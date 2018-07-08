//this version is 70 ms 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<cudnn.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<cublas_v2.h>
#include<string>
#include<fstream>
#include<cmath>
#include<ctime>

using namespace std;

void readWeights(float* weights, int m/*output*/, int n/*input*/, int h, int w, string baseFileName, bool readWeights = true) {


	string fileName = "weights2/" + baseFileName;
	if (readWeights) {
		fileName += "Weights.data";
	}
	else {
		fileName += "Biases.data";
	}
	ifstream in(fileName, ios::in | ios::binary);
	//cout << fileName << "\n";

	if (!in.is_open())

	{
		cout << "file " << baseFileName << "  didn't open \n";
		return;
	}
	in.read((char*)weights, m*n*h*w * sizeof(float));
	in.close();
	//cout << baseFileName << " :  " << weights[0] << "  " << weights[1] << "\n";

}


#define cudnnCheck(exp){\
cudnnStatus_t status=(exp);\
if(status!=CUDNN_STATUS_SUCCESS){\
std::cout<<"Error at line  "<<__LINE__<<"  "<<cudnnGetErrorString(status)<<"\n";\
std::exit(EXIT_FAILURE);\
}\
}\

#define cudaCheck(exp) {\
cudaError_t status=(exp);\
if(status!=cudaSuccess){\
cerr<<"error at cuda "<<__LINE__<<" "<<cudaGetErrorString(status)<<"\n";\
exit(EXIT_FAILURE);\
}\
}\

cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (image.empty()) { cerr << "couldn't open image\n"; }
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

	cv::Mat resizedImage(416, 416, CV_32FC2);
	cv::resize(image, resizedImage, cv::Size(416, 416), 0, 0, cv::INTER_CUBIC);
	if (resizedImage.empty())cerr << "resized image empty\n";
	//cout << "ok\n";
	return resizedImage;
}

void save_image(const char* output_filename, cv::Mat output_image) {
	//cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
	//cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	//output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

//incomplete
__global__ void leaky_relu_v2(float* d_data, float alpha, int size) {
	int index = (blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < size) {

		float x = d_data[index];
		if (x<0) d_data[index] = alpha*x;
	}
}
//try constant shift
__global__ void leaky_relu_v3(float* d_data, float alpha, int size, int step) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;

	if (index < step) {
		int channels = (size / step);

		index *= channels;
		for (int i = index; i < index + channels; i++) {
			float x = d_data[i];
			if (x<0) d_data[i] = alpha*x;
		}
	}
}
__global__ void leaky_relu_v4(float* d_data, float alpha, int size, int shift) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	index *= shift;
	if (index < size - shift) {

		for (int i = index; i < index + shift; i++) {
			float x = d_data[i];
			if (x<0) d_data[i] = alpha*x;
		}
	}
}
__global__ void leaky_relu(float* d_data, float alpha, int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {

		float x = d_data[index];
		if (x<0) d_data[index] = alpha*x;
	}
}
//step is width*height of the output of convolution
/*
@param size is width x height x channels
@Param step is width x height
the data in the format HxWxC
k is computed as index%(size/step)
*/
__global__ void add_biase(float* d_data, float* biases, int size/*WxHxC*/, int step/*WxH*/) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;

	if (index < step) {
		int biaseSize = (size / step);

		index *= biaseSize;
		for (int i = 0; i < biaseSize; i++) {
			d_data[index + i] += biases[i];

		}

	}
}

__global__ void add_biase_v2(float* d_data, float* biases, int size/*WxHxC*/, int channels) {

	int index = (blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if (index < size) {
		
		d_data[index] += biases[index % channels];

	}
}

__device__ float iou(float bx1x1, float bx1y1, float bx1x2, float bx1y2, float bx2x1, float bx2y1, float bx2x2, float bx2y2) {
	float x1 = (bx1x1 > bx2x1) ? bx1x1 : bx2x1;
	float y1 = (bx1y1> bx2y1) ? bx1y1 : bx2y1;
	float x2 = (bx1x2 > bx2x2) ? bx2x2 : bx1x2;
	float y2 = (bx1y2 > bx2y2) ? bx2y2 : bx1y2;
	float A1 = (bx1x2 - bx1x1)*(bx1y2 - bx1y1);
	float A2 = (bx2x2 - bx2x1)*(bx2y2 - bx2y1);
	float A_inter = ((x2 - x1) > 0 ? (x2 - x1) : 0)*((y2 - y1) > 0 ? (y2 - y1) : 0);
	return(A_inter / (A1 + A2 - A_inter));
}

//consider calculating the necessary points only
__global__ void calculate_points(float* boxes_dims, float* points, bool* boxes, int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;

	if (index < size) {
		//int left = h_boxes_dims[index] - (h_boxes_dims[index + 2] / 2.0);
		//int right = h_boxes_dims[index] + (h_boxes_dims[index + 2] / 2.0);
		//int top = h_boxes_dims[index + 1] - (h_boxes_dims[index + 3] / 2.0);
		//int bottom = h_boxes_dims[index + 1] + (h_boxes_dims[index + 3] / 2.0);
		int step = index * 4;
		float center_x = boxes_dims[step];
		float w = boxes_dims[step + 2];
		float center_y = boxes_dims[step + 1];
		float h = boxes_dims[step + 3];
		points[step] = center_x - ((w) / 2.0);
		points[step + 2] = center_x + ((w) / 2.0);
		points[step + 1] = center_y - ((h) / 2.0);
		points[step + 3] = center_y + ((h) / 2.0);

	}


}
__global__ void non_max_supression(float* points, bool* boxes, float* maxClassScore, int* maxClassIndex, float threashold = 0.3, int size = 13 * 13 * 5) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		float maxClass = maxClassScore[index];

		if (maxClass < 0.3) {
			boxes[index] = false;
			return;
		}

		int maxClassInd = maxClassIndex[index];

		float x1 = points[index * 4];
		float y1 = points[index * 4 + 1];
		float x2 = points[index * 4 + 2];
		float y2 = points[index * 4 + 3];
		for (int i = 0; i < size; i++) {

			if (boxes[i] && i != index) {
				if (maxClassInd == maxClassIndex[i]) {
					if (maxClass > maxClassScore[i]) {

						float x = iou(x1, y1, x2, y2, points[i * 4]
							, points[i * 4 + 1], points[i * 4 + 2], points[i * 4 + 3]);
						if (x >= threashold) {
							boxes[i] = false;

						}
					}
				}
			}
		}
	}

}

//20 classes
__global__ void exp(float* classes, int size) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x + threadIdx.x;
	if (index<size) {
		classes[index] = exp(classes[index]);
	}
}

__global__  void softmax(float* classes, int offset, float sum) {
	if (threadIdx.x < 20) {
		classes[threadIdx.x + offset] /= sum;
	}
}
__global__ void filter(float* classes, bool* boxes, float threshold = 0.4, int size = 13 * 13 * 5 * 20) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index < size) {
		if (classes[index] >= threshold) {
			boxes[index / 20] = true;
			//printf("index   %d   value   %f\n", index, classes[index]);
		}
	}
}
//blocks*threads
__global__ void sigmoid(float* x, int size) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x + threadIdx.x;
	if (index<size) {
		x[index] = 1 / (1 + exp(-1 * x[index]));
	}
}
//calculate centers of the box and the width and height
//calculate the necessary ones
__global__ void calculate_box_dims(float* x, float* d_anchors, int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		//center_x = (float(col) + sigmoid(tx)) * 32.0
		x[index] = (((index / (4)) % 13) + (1.0 / (1 + expf(-1 * x[index]))))*32.0;
		//center_y = (float(row) + sigmoid(ty)) * 32.0
		x[index + 1] = ((index / (13 * 4)) + (1.0 / (1 + expf(-1 * x[index + 1]))))*32.0;
		//roi_w = np.exp(tw) * anchors[2 * box + 0] * 32.0
		x[index + 2] = expf(x[index + 2])*d_anchors[2 * ((index / 25) % 5)] * 32.0;
		//roi_h = np.exp(th) * anchors[2 * box + 1] * 32.0
		x[index + 3] = expf(x[index + 3])*d_anchors[2 * ((index / 25) % 5) + 1] * 32.0;
	}
}

__global__ void sigmoid_exp(float* x, float* d_anchors, int size) {

	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index < size) {
		int cond = index % 25;
		switch (cond)
		{
		case 0:
			//center_x = (float(col) + sigmoid(tx)) * 32.0
			x[index] = (((index / (125)) % 13) + (1.0 / (1 + expf(-1 * x[index]))))*32.0;
			break;
		case 1:
			//center_y = (float(row) + sigmoid(ty)) * 32.0
			x[index] = ((index / (13 * 125)) + (1.0 / (1 + expf(-1 * x[index]))))*32.0;
			break;
		case 2:
			//roi_w = np.exp(tw) * anchors[2 * box + 0] * 32.0
			x[index] = expf(x[index])*d_anchors[2 * ((index / 25) % 5)] * 32.0;
			break;
		case 3:
			//roi_h = np.exp(th) * anchors[2 * box + 1] * 32.0
			x[index] = expf(x[index])*d_anchors[2 * ((index / 25) % 5) + 1] * 32.0;
			break;
		case 4:
			//confidence
			//if (index == 4)printf("data sample    %f\n\n", x[index]);
			x[index] = (1.0 / (1 + expf(-1 * x[index])));

			break;

		}
		//if (index <25)printf("data sample  %d   %f\n",index, x[index]);

	}
}

__global__ void scores(float* classes, float* confidence, int size) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		float x = confidence[index];
		int step = index * 20;
		for (int i = 0; i < 20; i++) {
			classes[step + i] *= x;
		}
	}
}

__global__ void get_max_scores(float* classes, bool* boxes, float* maxScores, int* maxIndecies, int size = 13 * 13 * 5) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;
	int classIndex = 20 * index;

	if (index < size) {

		if (boxes[index]) {
			float maxClassScore = classes[classIndex];
			int maxClassIndex = 0;

			float tmp = 0;
			for (int i = classIndex + 1; i < classIndex + 19; i++) {
				tmp = classes[i];
				if (tmp > maxClassScore) {
					maxClassScore = tmp;
					maxClassIndex = i - classIndex;
				}
			}
			//printf("from get_max_score		%d     %d\n", index,classIndex);
			maxScores[index] = maxClassScore;
			maxIndecies[index] = maxClassIndex;
		}


	}

}
__global__ void bool_arr(bool* d_boxes, int size, bool value = false) {
	int index = blockIdx.y*blockDim.x + blockIdx.x;
	if (index < size) {
		d_boxes[index] = value;
	}
}
__global__ void separate_data(float* predictions, float* boxes, float* confidence, float* classes, int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		int x = index % 25;
		if (x > 4) {

			classes[(index / 25) * 20 + (x - 5)] = predictions[index];
		}
		else if (x == 4)
		{
			confidence[(index / 25)] = predictions[index];
		}
		else
		{
			//centers and bounding boxes
			boxes[(index / 25) * 4 + x] = predictions[index];
		}
	}
}
//draw colored rectangles around objects 
//scale colors first
//thickness = 4 pixels
//size is WxH
__global__ void draw(float* d_image, int x1, int y1, int x2, int y2, float r, float g, float b, int w, int h, int thickness = 4) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	//scale for the three channels

	if (index < w*h) {
		//index *= 3;
		int xPos = (index / 3) % w;
		int yPos = (index / (3 * w));

		//on the same vertical line
		//increase x axis
		if ((yPos == y1 || yPos == y2) && (xPos >= x1 && xPos <= x2)) {
			for (int i = 0; i < thickness; i++) {
				if (index < w*h) {
					//r
					d_image[index] = 0;
					//g
					d_image[index + 1] = 0;
					//b
					d_image[index + 2] = 0;
					//next column ie next x in image terminology as the row here is column there 
					//remember image is at format  NHWC
					index += 3;
				}

			}
		}
		else if ((xPos == x1 || xPos == x2) && (yPos >= y1 && yPos <= y2))
		{
			for (int i = 0; i < thickness; i++) {
				if (index < w*h) {
					//r
					d_image[index] = 0;
					//g
					d_image[index + 1] = 0;
					//b
					d_image[index + 2] = 0;
				}
				index += (3 * h);
			}
		}

	}
}

template<class T>
void test(T* host_data, T* device_data, int start, int end) {
	cout << "host data \n\n";
	for (int i = start; i < end; i++) {
		cout << host_data[i] << "  ";
	}
	cout << "\n\n";

	T* tmp = (T*)malloc(end * sizeof(T));
	cudaMemcpy(tmp, device_data, end * sizeof(T), cudaMemcpyDeviceToHost);

	cout << "device data \n\n";
	for (int i = start; i < end; i++) {
		cout << tmp[i] << "  ";
	}
	cout << "\n\n";

}
template<class T>
void test(T* device_data, int start, int end) {

	T* tmp = (T*)malloc(end * sizeof(T));
	cudaCheck(cudaMemcpy(tmp, device_data, (end) * sizeof(T), cudaMemcpyDeviceToHost));

	cout << "device data \n\n";
	for (int i = start; i < end; i++) {
		cout << tmp[i] << "  ";
	}
	cout << "\n\n";
	//if (tmp[3] == true)cout << "True \n";
}

template<class T>
void test(T* device_data, int row, int col, int w, int step, int channels, int times, string name, int offset = 0, bool xDirection = true) {

	cout << name << "\n";
	for (int j = 0; j < times; j++) {
		test(device_data, (col*w*channels + row*channels + j*step + offset), (col*w*channels + row*channels + (j + 1)*step));
		//cout << (col*step*channels + row*channels + j*step + offset) <<"   "<< (col*step*channels + row*channels + (j + 1)*step) << "\n";
	}

}

//--------------------------------------things to be done for optimization---------------------------------------------------

//to be more memory effecient delete the unneeded values and re assign them 
// this maybe time costy
//test that 


//to be space effecient free workspace but make sure it doesn't include any data related to convolution


//make sure when it crashes because of memory to print that


//----------------------------------------------------------------------------------------------------------------------------


#define threadsPerBlock 32
#define shift 500
int main() {
	//  Layer        kernel    stride      output shape
	//	-------------------------------------------- -
	//Input(416,416,3)
	//	Convolution    3×3      1      (416, 416, 16)
	//	MaxPooling     2×2      2      (208, 208, 16)
	//	Convolution    3×3      1      (208, 208, 32)
	//	MaxPooling     2×2      2      (104, 104, 32)
	//	Convolution    3×3      1      (104, 104, 64)
	//	MaxPooling     2×2      2      (52, 52, 64)
	//	Convolution    3×3      1      (52, 52, 128)
	//	MaxPooling     2×2      2      (26, 26, 128)
	//	Convolution    3×3      1      (26, 26, 256)
	//	MaxPooling     2×2      2      (13, 13, 256)
	//	Convolution    3×3      1      (13, 13, 512)
	//	MaxPooling     2×2      1      (13, 13, 512)
	//	Convolution    3×3      1      (13, 13, 1024)
	//	Convolution    3×3      1      (13, 13, 1024)
	//	Convolution    1×1      1      (13, 13, 125)
	//	-------------------------------------------- -
	//all MAX POOLING is valid padding except last one but padding = 0 
	//all CONV are SAME padding with p = 1

	int imageH = 416, imageW = 416;
	float x = 1.0, y = 0.0;
	float* alpha = &x;
	float *beta = &y;

	long long totalSpace = 0;
	size_t space = 0;
	//std::cout << "ok\n";
	cudnnHandle_t cudnn;
	cudnnCheck(cudnnCreate(&cudnn));
	//input layer

	cudnnTensorDescriptor_t inputDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&inputDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(inputDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		3,
		imageH,
		imageW));


	//cv::Mat image = load_image("person.jpg");
	//std::cout << "image loaded with dims " << image.cols << " X " << image.rows << "\n";

	//for (int i = 0; i < 20; i++)std::cout << image.at<float>(cv::Point(0, i)) << "  ";
	//std::cout << "\n\n";

	float* d_input;
	cudaMalloc(&d_input, imageH*imageW * 3 * sizeof(float));

	totalSpace += imageH*imageW * 3 * sizeof(float);


	//load W1 
	float* w1 = (float*)malloc(16 * 3 * 3 * 3 * sizeof(float));
	readWeights(w1, 16, 3, 3, 3, "conv1");


	float* d_w1;
	cudaCheck(cudaMalloc(&d_w1, 16 * 3 * 3 * 3 * sizeof(float)));
	totalSpace += 16 * 3 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w1, w1, 16 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
	//(416, 416, 16)

	float* d_conv1Out;
	cudaCheck(cudaMalloc(&d_conv1Out, 16 * imageH * imageW * sizeof(float)));
	totalSpace += 16 * imageH * imageW * sizeof(float);
	//copy data to GPU

	//don't forget to add the biases

	float* b1 = (float*)malloc(16 * sizeof(float));
	readWeights(b1, 16, 1, 1, 1, "conv1", false);
	float* d_b1;
	cudaCheck(cudaMalloc(&d_b1, 16 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b1, b1, 16 * sizeof(float), cudaMemcpyHostToDevice));

	float* d_max1Out;
	cudaCheck(cudaMalloc(&d_max1Out, 208 * 208 * 16 * sizeof(float)));
	totalSpace += 208 * 208 * 16 * sizeof(float);


	//load W2 
	float* w2 = (float*)malloc(32 * 16 * 3 * 3 * sizeof(float));
	readWeights(w2, 32, 16, 3, 3, "conv2");

	float* d_w2;
	cudaCheck(cudaMalloc(&d_w2, 32 * 16 * 3 * 3 * sizeof(float)));
	totalSpace += 32 * 16 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w2, w2, 32 * 16 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

	float* d_conv2Out;
	cudaCheck(cudaMalloc(&d_conv2Out, 32 * 208 * 208 * sizeof(float)));
	totalSpace += 32 * 208 * 208 * sizeof(float);


	//don't forget to add the biases

	float* b2 = (float*)malloc(32 * sizeof(float));
	readWeights(b2, 32, 1, 1, 1, "conv2", false);

	float* d_b2;
	cudaCheck(cudaMalloc(&d_b2, 32 * sizeof(float)));
	cudaCheck(cudaMemcpy(d_b2, b2, 32 * sizeof(float), cudaMemcpyHostToDevice));

	//load W3
	float* w3 = (float*)malloc(64 * 32 * 3 * 3 * sizeof(float));
	readWeights(w3, 64, 32, 3, 3, "conv3");

	float* d_w3;
	cudaMalloc(&d_w3, 64 * 32 * 3 * 3 * sizeof(float));
	totalSpace += 64 * 32 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w3, w3, 64 * 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);


	float* b3 = (float*)malloc(64 * sizeof(float));
	readWeights(b3, 64, 1, 1, 1, "conv3", false);

	float* d_b3;
	cudaMalloc(&d_b3, 64 * sizeof(float));
	cudaMemcpy(d_b3, b3, 64 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_max3Out;
	cudaMalloc(&d_max3Out, 52 * 52 * 64 * sizeof(float));
	totalSpace += 52 * 52 * 64 * sizeof(float);

	//load W4
	float* w4 = (float*)malloc(128 * 64 * 3 * 3 * sizeof(float));
	readWeights(w4, 128, 64, 3, 3, "conv4");
	float* d_w4;
	cudaMalloc(&d_w4, 128 * 64 * 3 * 3 * sizeof(float));
	totalSpace += 128 * 64 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w4, w4, 128 * 64 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_conv4Out;
	cudaMalloc(&d_conv4Out, 128 * 52 * 52 * sizeof(float));
	totalSpace += 128 * 52 * 52 * sizeof(float);


	float* b4 = (float*)malloc(128 * sizeof(float));
	readWeights(b4, 128, 1, 1, 1, "conv4", false);

	float* d_b4;
	cudaMalloc(&d_b4, 128 * sizeof(float));


	cudaMemcpy(d_b4, b4, 128 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_max4Out;
	cudaMalloc(&d_max4Out, 26 * 26 * 128 * sizeof(float));
	totalSpace += 26 * 26 * 128 * sizeof(float);


	//load W5
	float* w5 = (float*)malloc(256 * 128 * 3 * 3 * sizeof(float));
	readWeights(w5, 256, 128, 3, 3, "conv5");
	float* d_w5;
	cudaMalloc(&d_w5, 256 * 128 * 3 * 3 * sizeof(float));
	totalSpace += 256 * 128 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w5, w5, 256 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_conv5Out;
	cudaMalloc(&d_conv5Out, 256 * 26 * 26 * sizeof(float));
	totalSpace += 256 * 26 * 26 * sizeof(float);

	float* b5 = (float*)malloc(256 * sizeof(float));
	readWeights(b5, 256, 1, 1, 1, "conv5", false);
	float* d_b5;
	cudaMalloc(&d_b5, 256 * sizeof(float));


	cudaMemcpy(d_b5, b5, 256 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_max5Out;
	cudaMalloc(&d_max5Out, 13 * 13 * 256 * sizeof(float));
	totalSpace += 13 * 13 * 256 * sizeof(float);

	//load W6
	float* w6 = (float*)malloc(512 * 256 * 3 * 3 * sizeof(float));
	readWeights(w6, 512, 256, 3, 3, "conv6");
	float* d_w6;
	cudaMalloc(&d_w6, 512 * 256 * 3 * 3 * sizeof(float));
	totalSpace += 512 * 256 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w6, w6, 512 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_conv6Out;
	cudaMalloc(&d_conv6Out, 512 * 13 * 13 * sizeof(float));
	totalSpace += 512 * 13 * 13 * sizeof(float);

	float* b6 = (float*)malloc(512 * sizeof(float));
	readWeights(b6, 512, 1, 1, 1, "conv6", false);
	float* d_b6;
	cudaMalloc(&d_b6, 512 * sizeof(float));


	cudaMemcpy(d_b6, b6, 512 * sizeof(float), cudaMemcpyHostToDevice);

	//here there's padding and stride 1
	float* d_max6Out;
	cudaMalloc(&d_max6Out, 13 * 13 * 512 * sizeof(float));
	totalSpace += 13 * 13 * 512 * sizeof(float);


	//load W7
	float* w7 = (float*)malloc(1024 * 512 * 3 * 3 * sizeof(float));
	readWeights(w7, 1024, 512, 3, 3, "conv7");
	float* d_w7;
	cudaMalloc(&d_w7, 1024 * 512 * 3 * 3 * sizeof(float));
	totalSpace += 1024 * 512 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w7, w7, 1024 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	float* d_conv7Out;
	cudaMalloc(&d_conv7Out, 1024 * 13 * 13 * sizeof(float));
	totalSpace += 1024 * 13 * 13 * sizeof(float);

	float* b7 = (float*)malloc(1024 * sizeof(float));
	readWeights(b7, 1024, 1, 1, 1, "conv7", false);
	float* d_b7;
	cudaCheck(cudaMalloc(&d_b7, 1024 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b7, b7, 1024 * sizeof(float), cudaMemcpyHostToDevice));

	//load W8
	float* w8 = (float*)malloc(1024 * 1024 * 3 * 3 * sizeof(float));
	readWeights(w8, 1024, 1024, 3, 3, "conv8", true);
	float* d_w8;
	cudaCheck(cudaMalloc(&d_w8, 1024 * 1024 * 3 * 3 * sizeof(float)));
	totalSpace += 1024 * 1024 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w8, w8, 1024 * 1024 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

	float* d_conv8Out;
	cudaCheck(cudaMalloc(&d_conv8Out, 1024 * 13 * 13 * sizeof(float)));
	totalSpace += 1024 * 13 * 13 * sizeof(float);


	float* b8 = (float*)malloc(1024 * sizeof(float));
	readWeights(b8, 1024, 1, 1, 1, "conv8", false);
	float* d_b8;
	cudaCheck(cudaMalloc(&d_b8, 1024 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b8, b8, 1024 * sizeof(float), cudaMemcpyHostToDevice));

	//load W9
	float* w9 = (float*)malloc(1024 * 125 * sizeof(float));
	readWeights(w9, 1024, 125, 3, 3, "conv9", true);
	float* d_w9;
	cudaCheck(cudaMalloc(&d_w9, 1024 * 125 * sizeof(float)));
	totalSpace += 1024 * 125 * sizeof(float);

	float* d_conv9Out;
	cudaCheck(cudaMalloc(&d_conv9Out, 125 * 13 * 13 * sizeof(float)));
	totalSpace += 125 * 13 * 13 * sizeof(float);
	cout << "total space " << totalSpace / (1024 * 1024) << " MB\n";


	float b9[125];
	readWeights(b9, 125, 1, 1, 1, "conv9", false);
	float* d_b9;
	cudaCheck(cudaMalloc(&d_b9, 125 * sizeof(float)));

	float* d_classes_softmax;
	cudaCheck(cudaMalloc(&d_classes_softmax, 13 * 13 * 5 * 20 * sizeof(float)));




	cv::Scalar colors[20] = { cv::Scalar(254.0, 254.0, 254),cv::Scalar(239.88888888888889, 211.66666666666669, 127),
		cv::Scalar(225.77777777777777, 169.33333333333334, 0), cv::Scalar(211.66666666666669, 127.0, 254),
		cv::Scalar(197.55555555555557, 84.66666666666667, 127), cv::Scalar(183.44444444444443, 42.33333333333332, 0),
		cv::Scalar(169.33333333333334, 0.0, 254), cv::Scalar(155.22222222222223, -42.33333333333335, 127),
		cv::Scalar(141.11111111111111, -84.66666666666664, 0), cv::Scalar(127.0, 254.0, 254),
		cv::Scalar(112.88888888888889, 211.66666666666669, 127), cv::Scalar(98.77777777777777, 169.33333333333334, 0),
		cv::Scalar(84.66666666666667, 127.0, 254), cv::Scalar(70.55555555555556, 84.66666666666667, 127),
		cv::Scalar(56.44444444444444, 42.33333333333332, 0), cv::Scalar(42.33333333333332, 0.0, 254),
		cv::Scalar(28.222222222222236, -42.33333333333335, 127), cv::Scalar(14.111111111111118, -84.66666666666664, 0),
		cv::Scalar(0.0, 254.0, 254), cv::Scalar(-14.111111111111118, 211.66666666666669, 127) };

	string classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse"
		, "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	//anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
	float h_anchors[10] = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };
	float* d_anchors;
	cudaCheck(cudaMalloc(&d_anchors, 10 * sizeof(float)));
	float* d_boxes_dims;
	cudaCheck(cudaMalloc(&d_boxes_dims, 13 * 13 * 5 * 4 * sizeof(float)));
	float* d_predictions;
	cudaCheck(cudaMalloc(&d_predictions, 13 * 13 * 5 * sizeof(float)));
	float* d_classes;
	cudaCheck(cudaMalloc(&d_classes, 13 * 13 * 5 * 20 * sizeof(float)));
	cudaCheck(cudaMemcpy(d_anchors, h_anchors, 10 * sizeof(float), cudaMemcpyHostToDevice));
	bool* d_boxes;
	cudaCheck(cudaMalloc(&d_boxes, 13 * 13 * 5 * sizeof(bool)));

	float* d_maxScorePerBox;
	cudaCheck(cudaMalloc(&d_maxScorePerBox, 13 * 13 * 5 * sizeof(float)));
	int* d_maxScoreIndex;
	cudaCheck(cudaMalloc(&d_maxScoreIndex, 13 * 13 * 5 * sizeof(int)));
	float* d_points;
	cudaCheck(cudaMalloc(&d_points, 13 * 13 * 5 * 4 * sizeof(float)));
	bool h_boxes[13 * 13 * 5];
	float* h_points = (float*)malloc(13 * 13 * 5 * 4 * sizeof(float));
	float h_maxScorePerBox[13 * 13 * 5];
	int h_maxScoreIndex[13 * 13 * 5];
	float* h_boxes_dims = (float*)malloc(13 * 13 * 5 * 4 * sizeof(float));


	cudaCheck(cudaMemcpy(d_b9, b9, 125 * sizeof(float), cudaMemcpyHostToDevice));


	//workspases
	void* workSpace[9] = { nullptr };

	//(16X3X3X3)
	cudnnFilterDescriptor_t w1Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w1Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w1Des,
		CUDNN_DATA_FLOAT,
		16,
		3,
		3,
		3));


	cudnnTensorDescriptor_t conv1OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv1OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv1OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		16,
		416,
		416));

	//cout << "output format NHWC \n";

	cudnnConvolutionDescriptor_t conv1Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv1Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv1Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv1Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		inputDes,
		w1Des,
		conv1Des,
		conv1OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv1Algo));



	cudnnTensorDescriptor_t max1OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max1OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max1OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		16,
		208,
		208));

	//cout << "max1 out NHWC\n";
	cudnnPoolingDescriptor_t max1Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max1Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max1Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		2,
		2));


	cudnnFilterDescriptor_t w2Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w2Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w2Des,
		CUDNN_DATA_FLOAT,
		32,
		16,
		3,
		3));

	//(208, 208, 32)
	cudnnTensorDescriptor_t conv2OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv2OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv2OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		32,
		208,
		208));

	//cout << "conv2 out NHWC\n";


	cudnnConvolutionDescriptor_t conv2Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv2Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv2Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv2Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max1OutDes,
		w2Des,
		conv2Des,
		conv2OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv2Algo));




	float* d_max2Out;
	cudaMalloc(&d_max2Out, 104 * 104 * 32 * sizeof(float));
	totalSpace += 104 * 104 * 32 * sizeof(float);
	cudnnTensorDescriptor_t max2OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max2OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max2OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		32,
		104,
		104));
	cudnnPoolingDescriptor_t max2Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max2Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max2Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		2,
		2));
	//[3,3,32,64]
	cudnnFilterDescriptor_t w3Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w3Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w3Des,
		CUDNN_DATA_FLOAT,
		64,
		32,
		3,
		3));


	//(104, 104, 64)
	cudnnTensorDescriptor_t conv3OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv3OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv3OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		64,
		104,
		104));

	float* d_conv3Out;
	cudaMalloc(&d_conv3Out, 64 * 104 * 104 * sizeof(float));
	totalSpace += 64 * 104 * 104 * sizeof(float);


	cudnnConvolutionDescriptor_t conv3Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv3Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv3Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv3Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max2OutDes,
		w3Des,
		conv3Des,
		conv3OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv3Algo));




	cudnnTensorDescriptor_t max3OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max3OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max3OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		64,
		52,
		52));
	cudnnPoolingDescriptor_t max3Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max3Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max3Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		2,
		2));

	cudnnFilterDescriptor_t w4Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w4Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w4Des,
		CUDNN_DATA_FLOAT,
		128,
		64,
		3,
		3));


	//(52, 52, 128)
	cudnnTensorDescriptor_t conv4OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv4OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv4OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		128,
		52,
		52));



	cudnnConvolutionDescriptor_t conv4Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv4Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv4Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv4Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max3OutDes,
		w4Des,
		conv4Des,
		conv4OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv4Algo));




	cudnnTensorDescriptor_t max4OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max4OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max4OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		128,
		26,
		26));
	cudnnPoolingDescriptor_t max4Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max4Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max4Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		2,
		2));

	//[3,3,128,256]
	cudnnFilterDescriptor_t w5Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w5Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w5Des,
		CUDNN_DATA_FLOAT,
		256,
		128,
		3,
		3));


	//(26, 26, 256)
	cudnnTensorDescriptor_t conv5OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv5OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv5OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		256,
		26,
		26));




	cudnnConvolutionDescriptor_t conv5Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv5Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv5Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv5Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max4OutDes,
		w5Des,
		conv5Des,
		conv5OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv5Algo));


	cudnnTensorDescriptor_t max5OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max5OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max5OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		256,
		13,
		13));
	cudnnPoolingDescriptor_t max5Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max5Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max5Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		2,
		2));

	cudnnFilterDescriptor_t w6Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w6Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w6Des,
		CUDNN_DATA_FLOAT,
		512,
		256,
		3,
		3));


	//(13, 13, 512)
	cudnnTensorDescriptor_t conv6OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv6OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv6OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		512,
		13,
		13));




	cudnnConvolutionDescriptor_t conv6Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv6Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv6Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv6Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max5OutDes,
		w6Des,
		conv6Des,
		conv6OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv6Algo));




	cudnnTensorDescriptor_t max6OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max6OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max6OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		512,
		13,
		13));
	cudnnPoolingDescriptor_t max6Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max6Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max6Des,
		CUDNN_POOLING_MAX,
		2,
		2,
		0,
		0,
		1,
		1));

	cudnnFilterDescriptor_t w7Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w7Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w7Des,
		CUDNN_DATA_FLOAT,
		1024,
		512,
		3,
		3));


	//(13 x  13 x 1024)
	cudnnTensorDescriptor_t conv7OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv7OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv7OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		1024,
		13,
		13));




	cudnnConvolutionDescriptor_t conv7Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv7Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv7Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv7Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max6OutDes,
		w7Des,
		conv7Des,
		conv7OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv7Algo));



	cudnnFilterDescriptor_t w8Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w8Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w8Des,
		CUDNN_DATA_FLOAT,
		1024,
		1024,
		3,
		3));


	//(13 x  13 x 1024)
	cudnnTensorDescriptor_t conv8OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv8OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv8OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		1024,
		13,
		13));




	cudnnConvolutionDescriptor_t conv8Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv8Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv8Des,
		1,
		1,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv8Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		conv7OutDes,
		w8Des,
		conv8Des,
		conv8OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv8Algo));



	//[1,1,1024,125]
	cudnnFilterDescriptor_t w9Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w9Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w9Des,
		CUDNN_DATA_FLOAT,
		125,
		1024,
		1,
		1));



	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w9, w9, 1024 * 125 * sizeof(float), cudaMemcpyHostToDevice));
	//(13 x  13 x 125)
	cudnnTensorDescriptor_t conv9OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv9OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv9OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		125,
		13,
		13));



	cudnnConvolutionDescriptor_t conv9Des;
	cudnnCheck(cudnnCreateConvolutionDescriptor(&conv9Des));
	cudnnCheck(cudnnSetConvolution2dDescriptor(conv9Des,
		0,
		0,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION));

	cudnnConvolutionFwdAlgo_t conv9Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		conv8OutDes,
		w9Des,
		conv9Des,
		conv9OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv9Algo));



	cudnnTensorDescriptor_t softmaxInputDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&softmaxInputDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(softmaxInputDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		5,
		20,
		13,
		13));
	cudnnTensorDescriptor_t softmaxOutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&softmaxOutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(softmaxOutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		5,
		20,
		13,
		13));
	int numBlocks[8] = { ceil(sqrt((416 * 416 * 16) / shift)) ,ceil(sqrt((208 * 208 * 32) / shift)) , ceil(sqrt((104 * 104 * 64) / shift))
		, ceil(sqrt((52 * 52 * 128) / shift)) , ceil(sqrt((26 * 26 * 256) / shift)) , ceil(sqrt((13 * 13 * 512) / shift))
		,ceil(sqrt((13 * 13 * 1024) / shift)) ,ceil(sqrt((13 * 13 * 1024) / shift)) };

	int numBlocksV2[9] = { ceil(sqrt((416 * 416 * 16) / threadsPerBlock)) ,ceil(sqrt((208 * 208 * 32) / threadsPerBlock)) 
		, ceil(sqrt((104 * 104 * 64) / threadsPerBlock))
		, ceil(sqrt((52 * 52 * 128) / threadsPerBlock)) , ceil(sqrt((26 * 26 * 256) / threadsPerBlock))
		, ceil(sqrt((13 * 13 * 512) / threadsPerBlock))
		,ceil(sqrt((13 * 13 * 1024) / threadsPerBlock)) ,ceil(sqrt((13 * 13 * 1024) / threadsPerBlock)),
		ceil(sqrt((13 * 13 * 125) / threadsPerBlock)) };
	//-------------------------------------------------------START------------------------------------------

	char* imagePaths[8] = { "dog.jpg","person.jpg","plane.jpg","motor.jpg","tv.jpg","horse.jpg" , "bus.jpg","bottle.jpg"};
	cv::Mat image[8];
	for (int i = 0; i < 8; i++) {
		image[i] = load_image(imagePaths[i]);

	}



	float* h_image = (float*)malloc(416 * 416 * 3 * sizeof(float));


	for (int i = 0; i < 8; i++) {
		long t1 = clock();
		cudaMemcpy(d_input, image[i].ptr<float>(0), imageH*imageW * 3 * sizeof(float), cudaMemcpyHostToDevice);
		std::cout << imagePaths[i] << "\n";
		//--------------------------------------------------------conv1----------------------------------------------------------

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			inputDes,
			w1Des,
			conv1Des,
			conv1OutDes,
			conv1Algo,
			&space));

		if (i == 0) {
			cudaCheck(cudaMalloc(&(workSpace[0]), space));
			totalSpace += space;
		}

		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			inputDes,
			d_input,
			w1Des,
			d_w1,
			conv1Des,
			conv1Algo,
			workSpace[0],
			space,
			beta,
			conv1OutDes,
			d_conv1Out));


		//add_biase << <dim3(416, 416), 1 >> >(d_conv1Out, d_b1, 416 * 416 * 16, 416 * 416);
		add_biase_v2 << <dim3(numBlocksV2[0],numBlocksV2[0]),threadsPerBlock >> > (d_conv1Out, d_b1, 416 * 416 * 16,16);

		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 1------------------------------------------------------------------
		//leaky_relu << <dim3(1665, 1665), 1 >> > (d_conv1Out, .1, 416 * 416 * 16);
		//int x = ceil(sqrt((416 * 416 * 16) / ( threadsPerBlock)));
		//std::cout << "x =  " << x << "\n";
		leaky_relu_v2 << < dim3(numBlocksV2[0], numBlocksV2[0]), threadsPerBlock  >> > (d_conv1Out, .1, 416 * 416 * 16);
		//leaky_relu_v3 << <dim3(416,416),1 >> > (d_conv1Out, .1, 416 * 416 * 16, 416 * 416);

		//leaky_relu_v4 << <dim3(numBlocks[0], numBlocks[0]), 1 >> > (d_conv1Out, .1, 416 * 416 * 16, shift);
		//----------------------------------------------------max 1----------------------------------------------------------------
		//	MaxPooling     2×2      2      (208, 208, 16)


		cudnnCheck(cudnnPoolingForward(cudnn,
			max1Des,
			alpha,
			conv1OutDes,
			d_conv1Out,
			beta,
			max1OutDes,
			d_max1Out));

		//--------------------------------------------------------conv2-------------------------------------------------------------------
		//[3,3,16,32]

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max1OutDes,
			w2Des,
			conv2Des,
			conv2OutDes,
			conv2Algo,
			&space));

		if (i == 0) {

			cudaCheck(cudaMalloc(&workSpace[1], space));
			totalSpace += space;
		}

		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max1OutDes,
			d_max1Out,
			w2Des,
			d_w2,
			conv2Des,
			conv2Algo,
			workSpace[1],
			space,
			beta,
			conv2OutDes,
			d_conv2Out));

		//add_biase << <dim3(208, 208), 1 >> >(d_conv2Out, d_b2, 208 * 208 * 32, 208 * 208);

		add_biase_v2 << <dim3(numBlocksV2[1], numBlocksV2[1]), threadsPerBlock >> > (d_conv2Out, d_b2, 208 * 208 * 32, 32);

		//	to be space effecient free workspace but make sure it doesn't include any data related to convolution


		//-----------------------------------------------------relu 2------------------------------------------------------------------
		//(208, 208, 32)
		//leaky_relu << <dim3(1180, 1180), 1 >> > (d_conv2Out, .1, 208 * 208 * 32);
		//leaky_relu_v3 << <dim3(208,208),1 >> > (d_conv2Out, .1, 208 * 208 * 32, 208 * 208);
		leaky_relu_v2 << <dim3(numBlocksV2[1], numBlocksV2[1]), threadsPerBlock >> > (d_conv2Out, .1, 208 * 208 * 32);
		//leaky_relu_v4 << <dim3(numBlocks[1], numBlocks[1]), 1 >> > (d_conv2Out, .1, 208 * 208 * 32, shift);
		//----------------------------------------------------max 2----------------------------------------------------------------
		//MaxPooling     2×2      2      (104, 104, 32)


		cudnnCheck(cudnnPoolingForward(cudnn,
			max2Des,
			alpha,
			conv2OutDes,
			d_conv2Out,
			beta,
			max2OutDes,
			d_max2Out));

		//--------------------------------------------------------conv3-------------------------------------------------------------------

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max2OutDes,
			w3Des,
			conv3Des,
			conv3OutDes,
			conv3Algo,
			&space));

		if (i == 0) {

			cudaMalloc(&workSpace[2], space);
			totalSpace += space;
		}


		long m = clock();
		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max2OutDes,
			d_max2Out,
			w3Des,
			d_w3,
			conv3Des,
			conv3Algo,
			workSpace[2],
			space,
			beta,
			conv3OutDes,
			d_conv3Out));
		cout << "time for conv 3 " << clock() - m << "\n";
		//don't forget to add the biases

		//add_biase << <dim3(104, 104), 1 >> >(d_conv3Out, d_b3, 104 * 104 * 64, 104 * 104);
		add_biase_v2 << <dim3(numBlocksV2[2], numBlocksV2[2]), threadsPerBlock >> > (d_conv3Out, d_b3, 104 * 104 * 64, 64);
		//-----------------------------------------------------relu 3------------------------------------------------------------------
		////(104, 104, 64)
		//leaky_relu << <dim3(835, 835), 1 >> > (d_conv3Out, .1, 104 * 104 * 64);
		//leaky_relu_v3 << <dim3(104, 104), 1 >> > (d_conv3Out, .1, 104 * 104 * 64, 104 * 104);
		leaky_relu_v2 << <dim3(numBlocksV2[2], numBlocksV2[2]), threadsPerBlock >> > (d_conv3Out, .1, 104 * 104 * 64);
		//leaky_relu_v4 << <dim3(numBlocks[2], numBlocks[2]), 1 >> > (d_conv3Out, .1, 104 * 104 * 64, shift);
		//----------------------------------------------------max 3----------------------------------------------------------------
		//MaxPooling     2×2      2      (52, 52, 64)



		cudnnCheck(cudnnPoolingForward(cudnn,
			max3Des,
			alpha,
			conv3OutDes,
			d_conv3Out,
			beta,
			max3OutDes,
			d_max3Out));

		//--------------------------------------------------------conv4-------------------------------------------------------------------
		//[3,3,64,128]

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max3OutDes,
			w4Des,
			conv4Des,
			conv4OutDes,
			conv4Algo,
			&space));


		if (i == 0) {
			cudaMalloc(&workSpace[3], space);
			totalSpace += space;
		}

		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max3OutDes,
			d_max3Out,
			w4Des,
			d_w4,
			conv4Des,
			conv4Algo,
			workSpace[3],
			space,
			beta,
			conv4OutDes,
			d_conv4Out));
		//don't forget to add the biases
		//cout << "time for conv 2 " << clock() - m << "\n";
		//add_biase << <dim3(52, 52), 1 >> >(d_conv4Out, d_b4, 52 * 52 * 128, 52 * 52);
		add_biase_v2 << <dim3(numBlocksV2[3], numBlocksV2[3]), threadsPerBlock >> > (d_conv4Out, d_b4, 52 * 52 * 128, 128);
		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 4------------------------------------------------------------------
		////(52, 52, 128)
		//leaky_relu << <dim3(600, 600), 1 >> > (d_conv4Out, .1, 52 * 52 * 128);
		//leaky_relu_v3 << <dim3(52, 52), 1 >> > (d_conv4Out, .1, 52 * 52 * 128, 52 * 52);
		leaky_relu_v2 << <dim3(numBlocksV2[3], numBlocksV2[3]), threadsPerBlock >> > (d_conv4Out, .1, 52 * 52 * 128);
		//leaky_relu_v4 << <dim3(numBlocks[3], numBlocks[3]), 1 >> > (d_conv4Out, .1, 52 * 52 * 128, shift);
		//----------------------------------------------------max 4----------------------------------------------------------------
		//MaxPooling     2×2      2      (26, 26, 128)



		cudnnCheck(cudnnPoolingForward(cudnn,
			max4Des,
			alpha,
			conv4OutDes,
			d_conv4Out,
			beta,
			max4OutDes,
			d_max4Out));
		//test(d_max4Out, 0, 16);
		//test(d_max4Out, 128, 128 + 16);
		////test(d_conv2Out, 32+16, 32 + 32);
		////test(d_conv1Out, 32 + 16, 32 + 32);
		//test(d_max4Out, 26 * 128, 26 * 128 + 16);
		//test(d_max4Out, 26 * 128 + 128, 26 * 128 + 128 + 16);
		//--------------------------------------------------------conv5-------------------------------------------------------------------


		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max4OutDes,
			w5Des,
			conv5Des,
			conv5OutDes,
			conv5Algo,
			&space));

		if (i == 0) {
			cudaMalloc(&workSpace[4], space);
			totalSpace += space;
		}


		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max4OutDes,
			d_max4Out,
			w5Des,
			d_w5,
			conv5Des,
			conv5Algo,
			workSpace[4],
			space,
			beta,
			conv5OutDes,
			d_conv5Out));
		//don't forget to add the biases

		//add_biase << <dim3(28, 28), 1 >> >(d_conv5Out, d_b5, 26 * 26 * 256, 26 * 26);
		add_biase_v2 << <dim3(numBlocksV2[4], numBlocksV2[4]), threadsPerBlock >> > (d_conv5Out, d_b5, 26 * 26 * 256, 256);
		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 5------------------------------------------------------------------
		////(26, 26, 256)
		//leaky_relu << <dim3(420, 420), 1 >> > (d_conv5Out, .1, 26 * 26 * 256);
		//leaky_relu_v3 << <dim3(26, 26), 1 >> > (d_conv5Out, .1, 26 * 26 * 256, 26 * 26);
		leaky_relu_v2 << <dim3(numBlocksV2[4], numBlocksV2[4]), threadsPerBlock >> > (d_conv5Out, .1, 26 * 26 * 256);
		//leaky_relu_v4 << <dim3(numBlocks[4], numBlocks[4]), 1 >> > (d_conv5Out, .1, 26 * 26 * 256, shift);
		//----------------------------------------------------max 5----------------------------------------------------------------
		//MaxPooling     2×2      2      (13, 13, 256)



		cudnnCheck(cudnnPoolingForward(cudnn,
			max5Des,
			alpha,
			conv5OutDes,
			d_conv5Out,
			beta,
			max5OutDes,
			d_max5Out));

		//--------------------------------------------------------conv6-------------------------------------------------------------------
		//[3,3,256,512]

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max5OutDes,
			w6Des,
			conv6Des,
			conv6OutDes,
			conv6Algo,
			&space));
		if (i == 0) {

			cudaMalloc(&workSpace[5], space);
			totalSpace += space;
		}


		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max5OutDes,
			d_max5Out,
			w6Des,
			d_w6,
			conv6Des,
			conv6Algo,
			workSpace[5],
			space,
			beta,
			conv6OutDes,
			d_conv6Out));
		//don't forget to add the biases

		//add_biase << <dim3(13, 13), 1 >> > (d_conv6Out, d_b6, 13 * 13 * 512, 13 * 13);
		add_biase_v2 << <dim3(numBlocksV2[5], numBlocksV2[5]), threadsPerBlock >> > (d_conv6Out, d_b6, 13 * 13 * 512, 512);
		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 6------------------------------------------------------------------
		////(13, 13, 512)
		//leaky_relu << <dim3(300, 300), 1 >> > (d_conv6Out, .1, 13 * 13 * 512);
		//leaky_relu_v3 << <dim3(13, 13), 1 >> > (d_conv6Out, .1, 13 * 13 * 512, 13 * 13);
		leaky_relu_v2 <<< dim3(numBlocksV2[5], numBlocksV2[5]), threadsPerBlock >> > (d_conv6Out, .1, 13 * 13 * 512);
		//leaky_relu_v4 << <dim3(numBlocks[5], numBlocks[5]), 1 >> > (d_conv6Out, .1, 13 * 13 * 512, shift);
		//----------------------------------------------------max 6----------------------------------------------------------------
		//MaxPooling     2×2      1      (13, 13, 512)



		cudnnCheck(cudnnPoolingForward(cudnn,
			max6Des,
			alpha,
			conv6OutDes,
			d_conv6Out,
			beta,
			max6OutDes,
			d_max6Out));

		//--------------------------------------------------------conv7-------------------------------------------------------------------
		//[3,3,512,1024]

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			max6OutDes,
			w7Des,
			conv7Des,
			conv7OutDes,
			conv7Algo,
			&space));
		if (i == 0) {


			cudaMalloc(&workSpace[6], space);
			totalSpace += space;
		}

		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			max6OutDes,
			d_max6Out,
			w7Des,
			d_w7,
			conv7Des,
			conv7Algo,
			workSpace[6],
			space,
			beta,
			conv7OutDes,
			d_conv7Out));
		//don't forget to add the biases



		//add_biase << <dim3(13, 13), 1 >> > (d_conv7Out, d_b7, 13 * 13 * 1024, 13 * 13);
		add_biase_v2 << <dim3(numBlocksV2[6], numBlocksV2[6]), threadsPerBlock >> > (d_conv7Out, d_b7, 13 * 13 * 1024, 1024);
		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 7------------------------------------------------------------------
		////(13 x  13 x 1024)
		//leaky_relu << <dim3(420, 420), 1 >> > (d_conv7Out, .1, 13 * 13 * 1024);
		//leaky_relu_v3 << <dim3(13, 13), 1 >> > (d_conv7Out, .1, 13 * 13 * 1024, 13 * 13);
		leaky_relu_v2 << <dim3(numBlocksV2[6], numBlocksV2[6]), threadsPerBlock >> > (d_conv7Out, .1, 13 * 13 * 1024);
		//leaky_relu_v4 << <dim3(numBlocks[6], numBlocks[6]), 1 >> > (d_conv7Out, .1, 13 * 13 * 1024, shift);
		//--------------------------------------------------------conv8-------------------------------------------------------------------
		//[3,3,1024,1024]


		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			conv7OutDes,
			w8Des,
			conv8Des,
			conv8OutDes,
			conv8Algo,
			&space));

		if (i == 0) {
			cudaMalloc(&workSpace[7], space);
			totalSpace += space;
		}

		//cout << "total space  " << totalSpace/(1024*1024) << "  MB\n";
		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			conv7OutDes,
			d_conv7Out,
			w8Des,
			d_w8,
			conv8Des,
			conv8Algo,
			workSpace[7],
			space,
			beta,
			conv8OutDes,
			d_conv8Out));
		//don't forget to add the biases

		//add_biase << <dim3(13, 13), 1 >> > (d_conv8Out, d_b8, 13 * 13 * 1024, 13 * 13);

		add_biase_v2 << <dim3(numBlocksV2[7], numBlocksV2[7]), threadsPerBlock >> > (d_conv8Out, d_b8, 13 * 13 * 1024, 1024);

		//to be space effecient free workspace but make sure it doesn't include any data related to convolution

		//-----------------------------------------------------relu 8------------------------------------------------------------------
		////(13 x  13 x 1024)
		//leaky_relu << <dim3(420, 420), 1 >> > (d_conv8Out, .1, 13 * 13 * 1024);
		//leaky_relu_v3 << <dim3(13, 13), 1 >> > (d_conv8Out, .1, 13 * 13 * 1024, 13 * 13);
		//x = ceil(sqrt((13 * 13 * 1024) / shift));
		leaky_relu_v2 << <dim3(numBlocksV2[7], numBlocksV2[7]), threadsPerBlock >> > (d_conv8Out, .1, 13 * 13 * 1024);
		//leaky_relu_v4 << <dim3(numBlocks[7], numBlocks[7]), 1 >> > (d_conv8Out, .1, 13 * 13 * 1024, shift);
		//--------------------------------------------------------conv9-------------------------------------------------------------------

		cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			conv8OutDes,
			w9Des,
			conv9Des,
			conv9OutDes,
			conv9Algo,
			&space));

		if (i == 0) {
			cudaMalloc(&workSpace[8], space);
			totalSpace += space;
		}

		cudnnCheck(cudnnConvolutionForward(cudnn,
			alpha,
			conv8OutDes,
			d_conv8Out,
			w9Des,
			d_w9,
			conv9Des,
			conv9Algo,
			workSpace[8],
			space,
			beta,
			conv9OutDes,
			d_conv9Out));
		//don't forget to add the biases

		//add_biase << <dim3(13, 13), 1 >> > (d_conv9Out, d_b9, 13 * 13 * 125, 13 * 13);
		//add_biase_v2 << <dim3(numBlocksV2[8], numBlocksV2[8]), threadsPerBlock >> > (d_conv9Out, d_b9, 13 * 13 * 125, 125);

		cudaDeviceSynchronize();
		cout << "time before softmax =  " <<clock() - t1 << "\n";

		//another optimization separate first then sigmoid exp  use the predefined ones
		sigmoid_exp << <dim3(150, 150), 1 >> > (d_conv9Out, d_anchors, 13 * 13 * 125);


		separate_data << <dim3(150, 150), 1 >> > (d_conv9Out, d_boxes_dims, d_predictions, d_classes, 13 * 13 * 125);



		cudnnCheck(cudnnSoftmaxForward(cudnn,
			CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			alpha,
			softmaxInputDes,
			d_classes,
			beta,
			softmaxOutDes,
			d_classes_softmax));
		cudaDeviceSynchronize();
		long t2 = clock();
		//cout << "time after softmax =  " << t2 - t1 << "\n";
		scores << <dim3(32, 32), 1 >> > (d_classes_softmax, d_predictions, 13 * 13 * 5);
		bool_arr << <dim3(30, 30), 1 >> >(d_boxes, 13 * 13 * 5, false);


		filter << < dim3(150, 150), 1 >> > (d_classes_softmax, d_boxes, 0.3, 13 * 13 * 5 * 20);

		get_max_scores << <dim3(30, 30), 1 >> > (d_classes_softmax, d_boxes, d_maxScorePerBox, d_maxScoreIndex, 13 * 13 * 5);

		calculate_points << <dim3(30, 30), 1 >> > (d_boxes_dims, d_points, d_boxes, 13 * 13 * 5);
		//cudaDeviceSynchronize();
		non_max_supression << < dim3(30, 30), 1 >> > (d_points, d_boxes, d_maxScorePerBox, d_maxScoreIndex, 0.3, 13 * 13 * 5);
		cudaDeviceSynchronize();
		long t3 = clock();
		//cout << "time before copy of data =  " << t3 - t2 << "\n";
		cudaCheck(cudaMemcpy(h_boxes, d_boxes, 13 * 13 * 5 * sizeof(bool), cudaMemcpyDeviceToHost));

		cudaCheck(cudaMemcpy(h_maxScorePerBox, d_maxScorePerBox, 13 * 13 * 5 * sizeof(float), cudaMemcpyDeviceToHost));

		cudaCheck(cudaMemcpy(h_maxScoreIndex, d_maxScoreIndex, 13 * 13 * 5 * sizeof(int), cudaMemcpyDeviceToHost));

		//cout << "time of copy of data =  " << clock() - t3 << "\n";
		//cudaCheck(cudaMemcpy(h_boxes_dims, d_boxes_dims, 13 * 13 * 5 * 4 * sizeof(float), cudaMemcpyDeviceToHost));


		cudaCheck(cudaMemcpy(h_points, d_points, 13 * 13 * 5 * 4 * sizeof(float), cudaMemcpyDeviceToHost));
		long t4 = clock();
		cv::Mat output(416, 416, CV_8UC3);
		cv::normalize(image[i], output, 0.0, 255.0, cv::NORM_MINMAX);
		for (int i = 0; i < 13 * 13 * 5; i++) {
			if (h_boxes[i]) {
				int index = i * 4;
				int left = h_points[index];
				int top = h_points[index + 1];
				int right = h_points[index + 2];
				int bottom = h_points[index + 3];
				float confidence = h_maxScorePerBox[i];
				string className = classes[h_maxScoreIndex[i]];
				std::cout << "( " << left << " , " << top << " ) , (" << right << " , " << bottom << " ) class "
					<< className << "  with prop  " << confidence  << "\n";

				//threashold boxes 
				left = (left <= 416) ? left : 416;
				top = (top <= 416) ? top : 416;
				right = (right <= 416) ? right : 416;
				bottom = (bottom <= 416) ? bottom : 416;
				cv::rectangle(output, cv::Point(left, top), cv::Point(right, bottom), colors[h_maxScoreIndex[i]], 3);
				//draw << <dim3(416, 416), 1 >> > (d_input, left, top, right, bottom, colors[h_maxScoreIndex[i]].val[0],
				//colors[h_maxScoreIndex[i]].val[1], colors[h_maxScoreIndex[i]].val[2], 416, 416);
				

			}

		}
		
		//cout << "time of post processing =  " << clock()-t4<< "\n";
		cudaCheck(cudaMemcpy(h_image, d_input, 416 * 416 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		cout << "total time =   " << clock() - t1 << "\n";
		//cv::Mat output0(416, 416, CV_32FC3,h_image);
		//cv::normalize(output0, output, 0.0, 255.0, cv::NORM_MINMAX);
		//cv::cvtColor(output, output, CV_RGB2BGR);
		//cv::normalize(output, output, 0.0, 255.0, cv::NORM_MINMAX);

		string num = std::to_string(i);
		string file = "output" + num + ".png";
		save_image(file.c_str(), output);
	}


	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	cout << "total space  " << totalSpace / (1024 * 1024) << "MB\n";



}
