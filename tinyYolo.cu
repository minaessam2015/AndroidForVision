
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

void readWeights(float* weights, int m/*output*/, int n/*input*/, int h, int w, string baseFileName,bool readWeights=true) {
	
	
	string fileName = "weights2/" + baseFileName;
	if (readWeights) {
		fileName+="Weights.data";
	}
	else {
		fileName += "Biases.data";
	}
	ifstream in(fileName, ios::in|ios::binary);
	//cout << fileName << "\n";

			if (!in.is_open())

			{
				cout << "file "<<baseFileName<<"  didn't open \n";
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
	cout << "ok\n";
	return resizedImage;
}

void save_image(const char* output_filename,cv::Mat output_image) {
	cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
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
c is computed as index%(size/step)
*/
__global__ void add_biase(float* d_data, float* biases, int size/*WxHxC*/, int step/*WxH*/) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		int k = index%(size/step);
		//printf("%d\n", k);
		d_data[index] += biases[k];
	}
}

__device__ float iou(float bx1x1,float bx1y1,float bx1x2,float bx1y2, float bx2x1, float bx2y1, float bx2x2, float bx2y2) {
	float x1 = (bx1x1 > bx2x1) ? bx1x1 : bx2x1;
	float y1 = (bx1y1> bx2y1) ? bx1y1 : bx2y1;
	float x2 = (bx1x2 > bx2x2) ? bx2x2 : bx1x2;
	float y2 = (bx1y2 > bx2y2) ? bx2y2 : bx1y2;
	float A1 = (bx1x2 - bx1x1)*(bx1y2 - bx1y1);
	float A2 = (bx2x2 - bx2x1)*(bx2y2 - bx2y1);
	float A_inter = ((x2 - x1) > 0 ? (x2 - x1) : 0)*((y2 - y1) > 0 ? (y2 - y1) : 0);
	return(A_inter / (A1 + A2 - A_inter));
}
__global__ void calculate_points(float* boxes_dims,float* points,bool* boxes,int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	
	if (index < size) {
		//int left = h_boxes_dims[index] - (h_boxes_dims[index + 2] / 2.0);
		//int right = h_boxes_dims[index] + (h_boxes_dims[index + 2] / 2.0);
		//int top = h_boxes_dims[index + 1] - (h_boxes_dims[index + 3] / 2.0);
		//int bottom = h_boxes_dims[index + 1] + (h_boxes_dims[index + 3] / 2.0);
		int step = index * 4;
		points[step] = boxes_dims[step] - ((boxes_dims[step + 2]) / 2.0);
		points[step+2]= boxes_dims[step] + ((boxes_dims[step + 2]) / 2.0);
		points[step + 1] = boxes_dims[step+1] - ((boxes_dims[step + 3]) / 2.0);
		points[step + 3] = boxes_dims[step+1] + ((boxes_dims[step + 3]) / 2.0);

	}
	

}
__global__ void non_max_supression(float* points, bool* boxes,float* maxClassScore, int* maxClassIndex,float threashold=0.3,int size=13*13*5) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		for (int i = 0; i < size; i++) {
			float maxClass=maxClassScore[index];
			int maxClassInd=maxClassIndex[index];
			if (boxes[i] && i != index) {
				if ( maxClassInd== maxClassIndex[i]) {
					if (maxClass > maxClassScore[i]) {
						float x = iou(points[index * 4], points[index * 4 + 1], points[index * 4 + 2], points[index * 4 + 3], points[i * 4]
							, points[i * 4 + 1], points[i * 4 + 2], points[i * 4 + 3]);
						if (x >= threashold) {
							boxes[i] = false;
							printf("from nms  %d   %f    %f \n", i, maxClass, x);
						}
					}
				}
			}
		}
	}

}

//20 classes
__global__ void exp(float* classes,int size) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index % 25 >= 5 && index<size) {
		classes[index] = exp(classes[index]);
	}
}

__global__  void softmax(float* classes,int offset, float sum) {
	if (threadIdx.x < 20) {
		classes[threadIdx.x + offset] /= sum;
	}
}
__global__ void filter(float* classes,bool* boxes,float threshold=0.4,int size=13*13*5*20) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index < size ) {
		if (classes[index] >= threshold) {
			boxes[index / 20] = true;
			printf("index   %d   value   %f\n", index, classes[index]);
		}
	}
}
__global__ void sigmoid(float* x,int size) {
	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index % 25 < 5&&index<size) {
		x[index] = 1 / (1 + exp(-1*x[index]));
	}
}

__global__ void sigmoid_exp(float* x,float* d_anchors, int size) {

	int index = (blockIdx.y*gridDim.x) + blockIdx.x;
	if (index < size) {
		int cond = index % 25;
		switch (cond)
		{
		case 0 :
		//center_x = (float(col) + sigmoid(tx)) * 32.0
			x[index] = (((index/(125))%13)+(1.0/(1+expf(-1*x[index]))))*32.0;
			break;
		case 1:
			//center_y = (float(row) + sigmoid(ty)) * 32.0
			x[index] = ((index/(13*125))  + (1.0 / (1 + expf(-1*x[index]))))*32.0;
			break;
		case 2 :
			//roi_w = np.exp(tw) * anchors[2 * box + 0] * 32.0
			x[index] = expf(x[index])*d_anchors[2 * ((index/25)%5)]*32.0;
			break;
		case 3 :
			//roi_h = np.exp(th) * anchors[2 * box + 1] * 32.0
			x[index] = expf(x[index])*d_anchors[2 * ((index / 25) % 5)+1]*32.0 ;
			break;
		case 4:
			//confidence
			//if (index == 4)printf("data sample    %f\n\n", x[index]);
			x[index] = (1.0 / (1 + expf(-1 * x[index])));
			
			break;

		default:
			//classes
			//x[index] = expf(x[index]);
			break;
		}
		//if (index <25)printf("data sample  %d   %f\n",index, x[index]);
		
	}
}

//try another implementation where u thread every thing
//use threads 20 
__global__ void scores(float* classes,float* confidence,int size) {
	
	int index = blockIdx.y*gridDim.x+blockIdx.x;
	if (index < size ) {
		float x = confidence[index];
		int step = index * 20;
		for (int i = 0; i < 20; i++) {
			classes[step + i] *= x;
		}
	}
}

__global__ void get_max_scores(float* classes, bool* boxes, float* maxScores , int* maxIndecies, int size=13*13*5) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;
	int classIndex = 20 * index ;
	
	if (index < size  ) {
		
		if (boxes[index]) {
			float maxClassScore = classes[classIndex];
			int maxClassIndex = 0;
			
			float tmp=0;
			for (int i=classIndex + 1 ; i < classIndex + 19; i++) {
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
__global__ void bool_arr (bool* d_boxes, int size, bool value=false) {
	int index = blockIdx.y*blockDim.x + blockIdx.x;
	if (index < size) {
		d_boxes[index] = value;
	}
}
__global__ void separate_data(float* predictions,float* boxes,float* confidence,float* classes,int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		int x = index % 25;
		if (x > 4) {

			classes[(index / 25)*20 + (x-5)] = predictions[index];
		}
		else if(x==4)
		{
			confidence[(index / 25)] = predictions[index];
		}
		else
		{
			//centers and bounding boxes
			boxes[(index / 25)*4 + x] = predictions[index];
		}
	}
}

template<class T>
void test(T* host_data,T* device_data,int start, int end) {
	cout << "host data \n\n";
	for (int i = start; i < end; i++) {
		cout << host_data[i] << "  ";
	}
	cout << "\n\n";

	T* tmp=(T*) malloc(end *sizeof(T));
	cudaMemcpy(tmp, device_data, end  * sizeof(T), cudaMemcpyDeviceToHost);

	cout << "device data \n\n";
	for (int i = start; i < end; i++) {
		cout << tmp[i] << "  ";
	}
	cout << "\n\n";

}
template<class T>
void test( T* device_data, int start , int end) {

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
void test(T* device_data,int row,int col,int w, int step, int channels,int times,string name,int offset=0,bool xDirection=true) {

	cout << name << "\n";
	for (int j = 0; j < times; j++) {
		test(device_data, (col*w*channels+row*channels+j*step+offset), (col*w*channels+row*channels + (j+1)*step));
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
	std::cout << "ok\n";
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

	
	cv::Mat image = load_image("person.jpg");
	std::cout << "image loaded with dims " << image.cols << " X " << image.rows << "\n";

	//for (int i = 0; i < 20; i++)std::cout << image.at<float>(cv::Point(0, i)) << "  ";
	//std::cout << "\n\n";

	float* d_input;
	cudaMalloc(&d_input, imageH*imageW * 3 * sizeof(float));
	cudaMemcpy(d_input, image.ptr<float>(0), imageH*imageW * 3 * sizeof(float), cudaMemcpyHostToDevice);
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

	float* b1=(float*)malloc(16*sizeof(float));
	readWeights(b1, 16, 1, 1, 1, "conv1",false);
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

	float* b2=(float*)malloc(32*sizeof(float));
	readWeights(b2, 32, 1, 1, 1, "conv2",false);

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


	float* b3=(float*) malloc(64*sizeof(float));
	readWeights(b3, 64, 1, 1, 1, "conv3",false);

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


	float* b4=(float*) malloc(128*sizeof(float));
	readWeights(b4, 128, 1, 1, 1, "conv4",false);

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

	float* b5=(float*)malloc(256*sizeof(float));
	readWeights(b5, 256, 1, 1, 1, "conv5",false);
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

	float* b6=(float*) malloc(512*sizeof(float));
	readWeights(b6, 512, 1, 1, 1, "conv6",false);
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

	float* b7=(float*) malloc(1024*sizeof(float));
	readWeights(b7, 1024, 1, 1, 1, "conv7",false);
	float* d_b7;
	cudaCheck(cudaMalloc(&d_b7, 1024 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b7, b7, 1024 * sizeof(float), cudaMemcpyHostToDevice));

	//load W8
	float* w8 = (float*)malloc(1024 * 1024 * 3 * 3 * sizeof(float));
	readWeights(w8, 1024, 1024, 3, 3, "conv8",true);
	float* d_w8;
	cudaCheck(cudaMalloc(&d_w8, 1024 * 1024 * 3 * 3 * sizeof(float)));
	totalSpace += 1024 * 1024 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w8, w8, 1024 * 1024 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

	float* d_conv8Out;
	cudaCheck(cudaMalloc(&d_conv8Out, 1024 * 13 * 13 * sizeof(float)));
	totalSpace += 1024 * 13 * 13 * sizeof(float);


	float* b8=(float*) malloc(1024*sizeof(float));
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


	cudaCheck(cudaMemcpy(d_b9, b9, 125 * sizeof(float), cudaMemcpyHostToDevice));

	//(16X3X3X3)
	cudnnFilterDescriptor_t w1Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w1Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w1Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

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
		CUDNN_PROPAGATE_NAN,
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
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

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
		CUDNN_PROPAGATE_NAN,
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
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv3Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max2OutDes,
		w3Des,
		conv3Des,
		conv3OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv3Algo));

	long t1 = clock();

	//read image
	//copy to GPU

	//--------------------------------------------------------conv1-------------------------------------------------------------------
	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		inputDes,
		w1Des,
		conv1Des,
		conv1OutDes,
		conv1Algo,
		&space));


	void* workSpace = nullptr;
	cudaCheck(cudaMalloc(&workSpace, space));
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		inputDes,
		d_input,
		w1Des,
		d_w1,
		conv1Des,
		conv1Algo,
		workSpace,
		space,
		beta,
		conv1OutDes,
		d_conv1Out));

	

	add_biase << <dim3(1665, 1665), 1 >> >(d_conv1Out, d_b1, 416 * 416 * 16, 416 * 416);
	//cout << "conv1 out \n";
	//test(d_conv1Out, 0, 16);
	//test(d_conv1Out, 16, 16 + 16);
	//test(d_conv1Out, 416 * 16, 416 * 16 + 16);
	//test(d_conv1Out, 416 * 16 + 16, 416 * 16 + 2 * 16);
	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 1------------------------------------------------------------------
	leaky_relu << <dim3(1665, 1665), 1 >> > (d_conv1Out, .1, 416 * 416 * 16);

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
	//std::cout << "max1 out \n\n";
	//test(d_max1Out, 0, 16);
	//test(d_max1Out, 16, 16 + 16);
	//test(d_max1Out, 416 * 16, 416 * 16 + 16);
	//test(d_max1Out, 416 * 16 + 16, 416 * 16 + 2 * 16);
	//cout << "total space " << totalSpace / (1024*1024) << "  MB\n";
	//--------------------------------------------------------conv2-------------------------------------------------------------------
	//[3,3,16,32]


	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max1OutDes,
		w2Des,
		conv2Des,
		conv2OutDes,
		conv2Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaCheck(cudaMalloc(&workSpace, space));
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max1OutDes,
		d_max1Out,
		w2Des,
		d_w2,
		conv2Des,
		conv2Algo,
		workSpace,
		space,
		beta,
		conv2OutDes,
		d_conv2Out));

	add_biase << <dim3(1180, 1180), 1 >> >(d_conv2Out, d_b2, 208 * 208 * 32, 208 * 208);
	//std::cout << "conv2 out \n";
	//test(d_conv2Out, 0, 16);
	//test(d_conv2Out, 32, 32 + 16);
	//test(d_conv2Out, 208 * 32, 208 * 32 + 16);
	//test(d_conv2Out, 208 * 32 + 32, 208 * 32 + 32 + 16);
//	to be space effecient free workspace but make sure it doesn't include any data related to convolution


	//-----------------------------------------------------relu 2------------------------------------------------------------------
	//(208, 208, 32)
	leaky_relu << <dim3(1180, 1180), 1 >> > (d_conv2Out, .1, 208 * 208 * 32);

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
	//std::cout << "max2 out \n";
	//test(d_max2Out, 0, 16);
	//test(d_max2Out, 32, 32 + 16);
	//test(d_max2Out, 104 * 32, 104 * 32 + 16);
	//test(d_max2Out, 104 * 32 + 32, 104 * 32 + 32 + 16);
	//--------------------------------------------------------conv3-------------------------------------------------------------------
	

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max2OutDes,
		w3Des,
		conv3Des,
		conv3OutDes,
		conv3Algo,
		&space));

	long m = clock();
	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max2OutDes,
		d_max2Out,
		w3Des,
		d_w3,
		conv3Des,
		conv3Algo,
		workSpace,
		space,
		beta,
		conv3OutDes,
		d_conv3Out));
	cout << "time for conv 3 " << clock() - m << "\n";
	//don't forget to add the biases



	add_biase << <dim3(835, 835), 1 >> >(d_conv3Out, d_b3, 104 * 104 * 64, 104 * 104);
	//to be space effecient free workspace but make sure it doesn't include any data related to convolution
	//std::cout << "conv3 out \n";
	//test(d_conv3Out, 0, 16);
	//test(d_conv3Out, 64, 64 + 16);
	//test(d_conv3Out, 104 * 64, 104 * 64 + 16);
	//test(d_conv3Out, 104 * 64 + 64, 104 * 64 + 64 + 16);

	//-----------------------------------------------------relu 3------------------------------------------------------------------
	////(104, 104, 64)
	leaky_relu << <dim3(835, 835), 1 >> > (d_conv3Out, .1, 104 * 104 * 64);

	//----------------------------------------------------max 3----------------------------------------------------------------
	//MaxPooling     2×2      2      (52, 52, 64)

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
		CUDNN_PROPAGATE_NAN,
		2,
		2,
		0,
		0,
		2,
		2));

	cudnnCheck(cudnnPoolingForward(cudnn,
		max3Des,
		alpha,
		conv3OutDes,
		d_conv3Out,
		beta,
		max3OutDes,
		d_max3Out));

	//std::cout << "max3 out \n";
	//test(d_max3Out, 0, 16);
	//test(d_max3Out, 64, 64 + 16);
	//test(d_max3Out, 52 * 64, 52 * 64 + 16);
	//test(d_max3Out, 52 * 64 + 64, 52 * 64 + 64 + 16);
	//--------------------------------------------------------conv4-------------------------------------------------------------------
	//[3,3,64,128]
	cudnnFilterDescriptor_t w4Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w4Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w4Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv4Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max3OutDes,
		w4Des,
		conv4Des,
		conv4OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv4Algo));
	
	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max3OutDes,
		w4Des,
		conv4Des,
		conv4OutDes,
		conv4Algo,
		&space));


	//void* workSpace = nullptr;
	//m = clock();
	//cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	//totalSpace += space;
	
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max3OutDes,
		d_max3Out,
		w4Des,
		d_w4,
		conv4Des,
		conv4Algo,
		workSpace,
		space,
		beta,
		conv4OutDes,
		d_conv4Out));
	//don't forget to add the biases
	//cout << "time for conv 2 " << clock() - m << "\n";

	add_biase << <dim3(600, 600), 1 >> >(d_conv4Out, d_b4, 52 * 52 * 128, 52 * 52);
	//cout << "conv 4 out \n";
	//test(d_conv4Out, 0, 16);
	//test(d_conv4Out, 128, 128 + 16);
	//test(d_conv4Out, 52 * 128, 52 * 128 + 16);
	//test(d_conv4Out, 52 * 128 + 128, 52 * 128 + 128 + 16);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 4------------------------------------------------------------------
	////(52, 52, 128)
	leaky_relu << <dim3(600, 600), 1 >> > (d_conv4Out, .1, 52 * 52 * 128);

	//----------------------------------------------------max 4----------------------------------------------------------------
	//MaxPooling     2×2      2      (26, 26, 128)

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
		CUDNN_PROPAGATE_NAN,
		2,
		2,
		0,
		0,
		2,
		2));

	cudnnCheck(cudnnPoolingForward(cudnn,
		max4Des,
		alpha,
		conv4OutDes,
		d_conv4Out,
		beta,
		max4OutDes,
		d_max4Out));


	//cout << "max4 out \n";
	//test(d_max4Out, 0, 16);
	//test(d_max4Out, 128, 128 + 16);
	//test(d_max4Out, 26 * 128, 26 * 128 + 16);
	//test(d_max4Out, 26 * 128+128, 26 * 128+128 + 16);
	//--------------------------------------------------------conv5-------------------------------------------------------------------
	//[3,3,128,256]
	cudnnFilterDescriptor_t w5Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w5Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w5Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv5Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max4OutDes,
		w5Des,
		conv5Des,
		conv5OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv5Algo));

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max4OutDes,
		w5Des,
		conv5Des,
		conv5OutDes,
		conv5Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max4OutDes,
		d_max4Out,
		w5Des,
		d_w5,
		conv5Des,
		conv5Algo,
		workSpace,
		space,
		beta,
		conv5OutDes,
		d_conv5Out));
	//don't forget to add the biases



	add_biase << <dim3(420, 420), 1 >> >(d_conv5Out, d_b5, 26 * 26 * 256, 26 * 26);

	//cout << "conv5  out \n";
	//test(d_conv5Out, 0, 16);
	//test(d_conv5Out, 256, 256 + 16);
	//test(d_conv5Out, 26 * 256, 26 * 256 + 16);
	//test(d_conv5Out, 26 * 256 + 256, 26 * 256 + 256 + 16);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 5------------------------------------------------------------------
	////(26, 26, 256)
	leaky_relu << <dim3(420, 420), 1 >> > (d_conv5Out, .1, 26 * 26 * 256);

	//----------------------------------------------------max 5----------------------------------------------------------------
	//MaxPooling     2×2      2      (13, 13, 256)

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
		CUDNN_PROPAGATE_NAN,
		2,
		2,
		0,
		0,
		2,
		2));

	cudnnCheck(cudnnPoolingForward(cudnn,
		max5Des,
		alpha,
		conv5OutDes,
		d_conv5Out,
		beta,
		max5OutDes,
		d_max5Out));

	//cout << "max5 out \n";
	//test(d_max5Out, 0, 16);
	//test(d_max5Out, 256, 256 + 16);
	//test(d_max5Out, 13 * 256, 13 * 256 + 16);
	//test(d_max5Out, 13 * 256 + 256, 13 * 256 + 256 + 16);

	//--------------------------------------------------------conv6-------------------------------------------------------------------
	//[3,3,256,512]
	cudnnFilterDescriptor_t w6Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w6Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w6Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv6Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max5OutDes,
		w6Des,
		conv6Des,
		conv6OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv6Algo));

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max5OutDes,
		w6Des,
		conv6Des,
		conv6OutDes,
		conv6Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max5OutDes,
		d_max5Out,
		w6Des,
		d_w6,
		conv6Des,
		conv6Algo,
		workSpace,
		space,
		beta,
		conv6OutDes,
		d_conv6Out));
	//don't forget to add the biases

	add_biase << <dim3(300, 300), 1 >> > (d_conv6Out, d_b6, 13 * 13 * 512, 13 * 13);

	//cout << "conv6 out \n";
	//test(d_conv6Out, 0, 16);
	//test(d_conv6Out, 512, 512 + 16);
	//test(d_conv6Out, 13 * 512, 13 * 512 + 16);
	//test(d_conv6Out, 13 * 512 + 512, 13 * 512 + 512 + 16);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 6------------------------------------------------------------------
	////(13, 13, 512)
	leaky_relu << <dim3(300, 300), 1 >> > (d_conv6Out, .1, 13 * 13 * 512);

	//cout << "relu6 out \n";
	//test(d_conv6Out, 0, 16);
	//test(d_conv6Out, 512, 512 + 16);
	//test(d_conv6Out, 13 * 512, 13 * 512 + 16);
	//test(d_conv6Out, 13 * 512 + 512, 13 * 512 + 512 + 16);
	//----------------------------------------------------max 6----------------------------------------------------------------
	//MaxPooling     2×2      1      (13, 13, 512)

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
		CUDNN_PROPAGATE_NAN,
		2,
		2,
		0,
		0,
		1,
		1));

	cudnnCheck(cudnnPoolingForward(cudnn,
		max6Des,
		alpha,
		conv6OutDes,
		d_conv6Out,
		beta,
		max6OutDes,
		d_max6Out));

	//cout << "max6 out \n";
	//test(d_max6Out, 0, 16);
	//test(d_max6Out, 512, 512 + 16);
	//test(d_max6Out, 13 * 512, 13 * 512 + 16);
	//test(d_max6Out, 13 * 512 + 512, 13 * 512 + 512 + 16);
	//--------------------------------------------------------conv7-------------------------------------------------------------------
	//[3,3,512,1024]
	cudnnFilterDescriptor_t w7Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w7Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w7Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv7Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max6OutDes,
		w7Des,
		conv7Des,
		conv7OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv7Algo));

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max6OutDes,
		w7Des,
		conv7Des,
		conv7OutDes,
		conv7Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max6OutDes,
		d_max6Out,
		w7Des,
		d_w7,
		conv7Des,
		conv7Algo,
		workSpace,
		space,
		beta,
		conv7OutDes,
		d_conv7Out));
	//don't forget to add the biases



	add_biase << <dim3(420, 420), 1 >> > (d_conv7Out, d_b7, 13 * 13 * 1024, 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 7------------------------------------------------------------------
	////(13 x  13 x 1024)
	leaky_relu << <dim3(420, 420), 1 >> > (d_conv7Out, .1, 13 * 13 * 1024);


	//--------------------------------------------------------conv8-------------------------------------------------------------------
	//[3,3,1024,1024]
	cudnnFilterDescriptor_t w8Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w8Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w8Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv8Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		conv7OutDes,
		w8Des,
		conv8Des,
		conv8OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv8Algo));

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		conv7OutDes,
		w8Des,
		conv8Des,
		conv8OutDes,
		conv8Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	//cout << "total space  " << totalSpace/(1024*1024) << "  MB\n";
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		conv7OutDes,
		d_conv7Out,
		w8Des,
		d_w8,
		conv8Des,
		conv8Algo,
		workSpace,
		space,
		beta,
		conv8OutDes,
		d_conv8Out));
	//don't forget to add the biases



	add_biase << <dim3(420, 420), 1 >> > (d_conv8Out, d_b8, 13 * 13 * 1024, 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 8------------------------------------------------------------------
	////(13 x  13 x 1024)
	leaky_relu << <dim3(420, 420), 1 >> > (d_conv8Out, .1, 13 * 13 * 1024);

	//test(d_conv8Out, 0,20);

	//--------------------------------------------------------conv9-------------------------------------------------------------------
	//[1,1,1024,125]
	cudnnFilterDescriptor_t w9Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w9Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w9Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t conv9Algo;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		conv8OutDes,
		w9Des,
		conv9Des,
		conv9OutDes,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv9Algo));

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		conv8OutDes,
		w9Des,
		conv9Des,
		conv9OutDes,
		conv9Algo,
		&space));


	//void* workSpace = nullptr;
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		conv8OutDes,
		d_conv8Out,
		w9Des,
		d_w9,
		conv9Des,
		conv9Algo,
		workSpace,
		space,
		beta,
		conv9OutDes,
		d_conv9Out));
	//don't forget to add the biases

	add_biase << <dim3(150, 150), 1 >> > (d_conv9Out, d_b9, 13 * 13 * 125, 13 * 13);

	////test
	//cudaDeviceSynchronize();

	//cout << "predictions\n\n";
	//test(d_conv9Out, 0, 25);
	//test(d_conv9Out, 25, 25 + 25);
	//test(d_conv9Out, 50, 75);
	//test(d_conv9Out, 75, 100);

	//anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
	float h_anchors[10] = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };
	float* d_anchors;
	cudaCheck(cudaMalloc(&d_anchors, 10 * sizeof(float)));
	cudaCheck(cudaMemcpy(d_anchors, h_anchors, 10 * sizeof(float), cudaMemcpyHostToDevice));

	sigmoid_exp << <dim3(150,150),1 >> > (d_conv9Out,d_anchors, 13 * 13 * 125);
	//cout << "confidence and centeres \n";
	//test(d_conv9Out, 125, 130);
	//test(d_conv9Out, 150, 150 + 5);
	//test(d_conv9Out, 175, 180);
	//test(d_conv9Out, 200, 205);
	//cudaDeviceSynchronize();
	float* d_boxes_dims;
	cudaCheck(cudaMalloc(&d_boxes_dims, 13 * 13 * 5 * 4 * sizeof(float)));
	float* d_predictions;
	cudaCheck(cudaMalloc(&d_predictions, 13 * 13 * 5 * sizeof(float)));
	float* d_classes;
	cudaCheck(cudaMalloc(&d_classes, 13 * 13 * 5 * 20 * sizeof(float)));

	separate_data << <dim3(150, 150), 1 >> > (d_conv9Out, d_boxes_dims, d_predictions, d_classes, 13 * 13 * 125);

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

	float* d_classes_softmax;
	cudaCheck(cudaMalloc(&d_classes_softmax, 13 * 13 * 5 * 20 * sizeof(float)));


	cudnnCheck(cudnnSoftmaxForward(cudnn,
		CUDNN_SOFTMAX_FAST,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		alpha,
		softmaxInputDes,
		d_classes,
		beta,
		softmaxOutDes,
		d_classes_softmax));


	//for (int i = 0; i < 10; i++) {
	//	cout << "after separation\n";
	//	test(d_boxes_dims, 4 * i, 4 * i + 4);
	//	test(d_predictions, i, i + 1);
	//	test(d_classes_softmax, 20 * i, 20 * i + 20);
	//}


	//float  h_conv9Out[13 * 13 * 125];
	//cudaCheck(cudaMemcpy(h_conv9Out, d_conv9Out, 13 * 13 * 125 * sizeof(float), cudaMemcpyDeviceToHost));
	//for (int i = 5; i < 13 * 13 * 125; i+=25) {
	//	float sum = 0.0;
	//	for (int j = i; j < i + 20; j++) {
	//		sum += h_conv9Out[j];
	//	}
	//	softmax<<<dim3(1,1),20>>>(d_conv9Out, i, sum);
	//}
	//cout << "after softmax\n";

	//test(d_conv9Out, 13*125+5, 13 * 125 + 5+20);
	//test(d_conv9Out, 13 * 125 + 5+25, 13 * 125 + 5 + 25 + 20);
	//test(d_conv9Out, 13 * 125 + 5 + 25, 13 * 125 + 5 + 25 + 20);
	//test(d_conv9Out, 205, 225);
	//test(d_conv9Out, 2, 2,13, 25, 125, 5, "softmax and centers",0);

	//test<float>(d_predictions, 0, 13*13*5);

	//test(d_predictions, 176, 177);
	//test<float>(d_boxes_dims, 176 * 4, 176 * 4 + 4);
	//test<float>(d_classes_softmax, 176 * 20, 176 * 20 + 20);

	//test(d_predictions, 536, 537);
	//test<float>(d_boxes_dims, 536 * 4, 536 * 4 + 4);
	//test<float>(d_classes_softmax, 536 * 20, 536 * 20 + 20);

	scores << <dim3(32,32),1 >> > (d_classes_softmax,d_predictions,13*13*5);
	//cudaDeviceSynchronize();
	float* h_predictions = (float*)malloc(13 * 13 * 5 *20* sizeof(float));
	cudaCheck(cudaMemcpy(h_predictions, d_classes_softmax, 13 * 13 * 5 *20* sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 13 * 13 * 5*20; i++) {
		if (h_predictions[i] > .3)cout << "index    " << i << "   value   " << h_predictions[i] << "   \n";
	}
	cout << "\n\n";

	//test(d_predictions, 176, 177);
	//test<float>(d_boxes_dims, 176 * 4, 176 * 4 + 4);
	//test<float>(d_classes_softmax, 176 * 20, 176 * 20 + 20);

	//test(d_predictions, 536, 537);
	//test<float>(d_boxes_dims, 536 * 4, 536 * 4 + 4);
	//test<float>(d_classes_softmax, 536 * 20, 536 * 20 + 20);


	
	//initialize the array on GPU
	bool* d_boxes;
	
	cudaCheck(cudaMalloc(&d_boxes,13*13*5*sizeof(bool)));
	bool_arr<<<dim3(30,30),1>>>(d_boxes, 13 * 13 * 5, false);

	filter <<< dim3(150,150),1>> > (d_classes_softmax,d_boxes, 0.3 , 13*13*5*20);

	float* d_maxScorePerBox;
	cudaCheck(cudaMalloc(&d_maxScorePerBox, 13 * 13 * 5 * sizeof(float)));
	int* d_maxScoreIndex;
	cudaCheck(cudaMalloc(&d_maxScoreIndex, 13 * 13 * 5 * sizeof(int)));

	get_max_scores << <dim3(30, 30), 1 >> > (d_classes_softmax, d_boxes, d_maxScorePerBox, d_maxScoreIndex, 13 * 13 * 5);

	float* d_points;
	cudaCheck(cudaMalloc(&d_points, 13 * 13 * 5 * 4 * sizeof(float)));

	calculate_points << <dim3(30, 30), 1 >> > (d_boxes_dims, d_points, d_boxes, 13 * 13 * 5);
	//cudaDeviceSynchronize();
	non_max_supression << < dim3(30,30),1>> > (d_points,d_boxes,d_maxScorePerBox,d_maxScoreIndex,0.3,13*13*5);

	string classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse"
		, "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
	//extract incomplete info 
	bool h_boxes[13 * 13 * 5];
	cudaCheck(cudaMemcpy(h_boxes, d_boxes, 13 * 13 * 5 * sizeof(bool), cudaMemcpyDeviceToHost));
	float h_maxScorePerBox[13 * 13 * 5];
	cudaCheck(cudaMemcpy(h_maxScorePerBox,d_maxScorePerBox, 13 * 13 * 5 * sizeof(float),cudaMemcpyDeviceToHost));
	int h_maxScoreIndex[13 * 13 * 5 ];
	cudaCheck(cudaMemcpy(h_maxScoreIndex, d_maxScoreIndex, 13 * 13 * 5 * sizeof(int), cudaMemcpyDeviceToHost));

	float* h_boxes_dims=(float*) malloc(13*13*5*4*sizeof(float));
	cudaCheck(cudaMemcpy(h_boxes_dims, d_boxes_dims, 13 * 13 * 5 * 4 * sizeof(float), cudaMemcpyDeviceToHost));

	float* h_points = (float*)malloc(13 * 13 * 5 * 4 * sizeof(float));
	cudaCheck(cudaMemcpy(h_points, d_points, 13 * 13 * 5 * 4 * sizeof(float), cudaMemcpyDeviceToHost));

	cv::Scalar colors[20] = { cv::Scalar (254.0, 254.0, 254),cv::Scalar(239.88888888888889, 211.66666666666669, 127),
		cv::Scalar (225.77777777777777, 169.33333333333334, 0), cv::Scalar (211.66666666666669, 127.0, 254),
		cv::Scalar (197.55555555555557, 84.66666666666667, 127), cv::Scalar (183.44444444444443, 42.33333333333332, 0),
		cv::Scalar (169.33333333333334, 0.0, 254), cv::Scalar (155.22222222222223, -42.33333333333335, 127),
		cv::Scalar (141.11111111111111, -84.66666666666664, 0), cv::Scalar (127.0, 254.0, 254),
		cv::Scalar (112.88888888888889, 211.66666666666669, 127), cv::Scalar (98.77777777777777, 169.33333333333334, 0),
		cv::Scalar (84.66666666666667, 127.0, 254), cv::Scalar (70.55555555555556, 84.66666666666667, 127),
		cv::Scalar (56.44444444444444, 42.33333333333332, 0), cv::Scalar (42.33333333333332, 0.0, 254),
		cv::Scalar (28.222222222222236, -42.33333333333335, 127), cv::Scalar (14.111111111111118, -84.66666666666664, 0),
		cv::Scalar (0.0, 254.0, 254), cv::Scalar (-14.111111111111118, 211.66666666666669, 127) };

	cv::Mat output(416, 416, CV_32FC3);
	image.copyTo(output);
	//cout << output.at<float>(0, 0) << "  " <<  output.at<float>(0, 1) << "\n";
	for (int i = 0; i < 13 * 13 * 5;i++) {
		if (h_boxes[i]) {
			int index = i * 4;
			// Compute the final coordinates on both axes
			//int left = h_boxes_dims[index] - (h_boxes_dims[index + 2] / 2.0);
			//int right = h_boxes_dims[index] + (h_boxes_dims[index + 2] / 2.0);
			//int top = h_boxes_dims[index+1] - (h_boxes_dims[index + 3] / 2.0);
			//int bottom = h_boxes_dims[index + 1] + (h_boxes_dims[index + 3] / 2.0);
			int left = h_points[index];
			int top = h_points[index+1];
			int right = h_points[index+2];
			int bottom = h_points[index + 3];

			std::cout << "( " << left << " , " << top << " ) , (" << right << " , " << bottom << " ) class "
				<<classes[h_maxScoreIndex[i]]<<"  with prop  "<<h_maxScorePerBox[i]<<"\n";
			if (left < 416 && top < 416 && right < 416 && bottom < 416) {
				cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), colors[h_maxScoreIndex[i]]);
			}
				

		}
	}

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution
	long t2 = clock();
	cout << "total space  " << totalSpace / (1024 * 1024) << "MB\n";
	cout << "time =  " << t2 - t1 << "\n";
	save_image("output.png", image);

}
