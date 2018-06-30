
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
template<int channels,int height,int width>
void readWeights(float weights[][channels][height][width], int m/*output*/, int n/*input*/, int h, int w, string baseFileName) {
	cout << channels*height*width << "\n";
	return;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			string fileName = "weights/" + baseFileName + std::to_string(j) + "X" + std::to_string(i) + ".txt";
			ifstream in(fileName, std::ifstream::in);
			//cout << fileName << "\n";

			char c;
			if (!in.is_open())

			{
				cout << "file didn't open \n";
				return;
			}
			string s = "";
			for (int k = 0; k < h; k++) {
				//cout << s.length()<<"\n";
				for (int l = 0; l < w; l++) {
					//cout << "L " << l;
					while (in.get(c)) {
						if (c == ' '&&s.length() == 0)continue;
						if (c != ' '&&c != '\n')s += c;
						else
						{
							break;
						}
						//cout << c << " ";
					}
					if (s.length()>0)weights[i][j][k][l] = std::stof(s);
					s = "";

				}
			}
		}
	}

}


#define cudnnCheck(exp){\
cudnnStatus_t status=(exp);\
if(status!=CUDNN_STATUS_SUCCESS){\
std::cout<<"Error at line  "<<__LINE__<<"  "<<cudnnGetErrorString(status)<<"\n";\
std::exit(EXIT_FAILURE);\
}\
}\

#define cudaCheck(exp){\
cudaError_t status=(exp);\
if(status!=cudaSuccess){\
cerr<<"error at cuda "<<__LINE__<<" "<<cudaGetErrorString(status)<<"\n";\
}\
}\

cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (image.empty()) { cerr << "couldn't open image\n"; }
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	cout << "ok\n";
	cv::Mat resizedImage(416,416,CV_32FC2);
	cv::resize(image, resizedImage, cv::Size(416, 416), 0, 0, cv::INTER_CUBIC);
	if (resizedImage.empty())cerr << "resized image empty\n";
	cout << "ok\n";
	return resizedImage;
}

__global__ void leaky_relu(float* d_data,float alpha,int size) {
	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {

		float x = d_data[index];
		if(x<0) d_data[index]=alpha*x;
	}
}
//step is width*height of the output of convolution
__global__ void add_biase(float* d_data, float* biases,int size, int step) {

	int index = blockIdx.y*gridDim.x + blockIdx.x;
	if (index < size) {
		int k = int(index / step);
		//printf("%d\n", k);
		d_data[index] += biases[k];
	}
}

//--------------------------------------things to be done for optimization---------------------------------------------------

//to be more memory effecient delete the unneeded values and re assign them 
// this maybe time costy
//test that 


//to be space effecient free workspace but make sure it doesn't include any data related to convolution


//make sure when it crashes because of memory to print that


//----------------------------------------------------------------------------------------------------------------------------

int main(){
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
	//all MAX POOLING is valid padding except last one
	//all CONV are SAME padding with p = 1

	int imageH = 416, imageW=416;
	float x = 1.0, y = 0.0;
	float* alpha = &x;
	float *beta=&y;

	size_t totalSpace = 0;
	size_t space = 0;
	cout << "ok\n";
	cv::Mat image=load_image("dog.jpg");
	cout << "image loaded with dims " << image.cols << " X " << image.rows << "\n";
	float* d_input;
	cudaMalloc(&d_input, imageH*imageW * 3 * sizeof(float));
	cudaMemcpy(d_input, image.ptr<float>(0), imageH*imageW * 3 * sizeof(float), cudaMemcpyHostToDevice);
	totalSpace += imageH*imageW * 3 * sizeof(float);
	
	long t1 = clock();
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

	//read image
	//copy to GPU

	//--------------------------------------------------------conv1-------------------------------------------------------------------
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

	//load W1 
	float* w1=(float*)malloc(16 * 3 * 3 * 3 * sizeof(float));
	float* d_w1;
	cudaCheck(cudaMalloc(&d_w1, 16 * 3 * 3 * 3 * sizeof(float)));
	totalSpace += 16 * 3 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w1, w1, 16 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
	//(416, 416, 16)
	cudnnTensorDescriptor_t conv1OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv1OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv1OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		16,
		416,
		416));

	float* d_conv1Out;
	cudaCheck(cudaMalloc(&d_conv1Out, 16 * imageH * imageW  * sizeof(float)));
	totalSpace += 16 * imageH * imageW  * sizeof(float);
	//copy data to GPU
	
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
	//don't forget to add the biases

	float b1[16] = { 0 };
	float* d_b1;
	cudaCheck(cudaMalloc(&d_b1, 16 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b1, b1, 16 * sizeof(float), cudaMemcpyHostToDevice));

	add_biase<<<dim3(1665, 1665),1>>>(d_conv1Out, d_b1, 416 * 416 * 16 , 416 * 416);
	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 1------------------------------------------------------------------
	leaky_relu <<<dim3(1665, 1665), 1 >>> (d_conv1Out, .01, 416 * 416 * 16);

	//----------------------------------------------------max 1----------------------------------------------------------------
	//	MaxPooling     2×2      2      (208, 208, 16)
	float* d_max1Out;
	cudaCheck(cudaMalloc(&d_max1Out, 208 * 208 * 16 * sizeof(float)));
	totalSpace += 208 * 208 * 16 * sizeof(float);
	cudnnTensorDescriptor_t max1OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max1OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max1OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		16,
		208,
		208));
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

	cudnnCheck(cudnnPoolingForward(cudnn,
		max1Des,
		alpha,
		conv1OutDes,
		d_conv1Out,
		beta,
		max1OutDes,
		d_max1Out));
	cout << "total space " << totalSpace / (1024*1024) << "  MB\n";
	//--------------------------------------------------------conv2-------------------------------------------------------------------
	//[3,3,16,32]
	cudnnFilterDescriptor_t w2Des;
	cudnnCheck(cudnnCreateFilterDescriptor(&w2Des));
	cudnnCheck(cudnnSetFilter4dDescriptor(w2Des,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		32,
		16,
		3,
		3));

	//load W2 
	float* w2=(float*)malloc(32 * 16 * 3 * 3 * sizeof(float));
	float* d_w2;
	cudaCheck(cudaMalloc(&d_w2, 32 * 16 * 3 * 3 * sizeof(float)));
	totalSpace += 32 * 16 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w2, w2, 32 * 16 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
	//(208, 208, 32)
	cudnnTensorDescriptor_t conv2OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv2OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv2OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		32,
		208,
		208));

	float* d_conv2Out;
	cudaCheck(cudaMalloc(&d_conv2Out, 32 * 208 * 208 * sizeof(float)));
	totalSpace +=  32 * 208 * 208  * sizeof(float);
	

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
	//don't forget to add the biases

	float b2[32] = { 0 };
	float* d_b2;
	cudaCheck(cudaMalloc(&d_b2, 32 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b2, b2, 32 * sizeof(float), cudaMemcpyHostToDevice));

	add_biase << <dim3(1180, 1180), 1 >> >(d_conv2Out, d_b2, 208 * 208 * 32 , 208 * 208);
	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 2------------------------------------------------------------------
	//(208, 208, 32)
	leaky_relu << <dim3(1180, 1180), 1 >> > (d_conv2Out, .01, 208 * 208 * 32);

	//----------------------------------------------------max 2----------------------------------------------------------------
	//MaxPooling     2×2      2      (104, 104, 32)
	float* d_max2Out;
	cudaMalloc(&d_max2Out, 104 * 104 * 32 * sizeof(float));
	totalSpace += 104 * 104 * 32 * sizeof(float);
	cudnnTensorDescriptor_t max2OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max2OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max2OutDes,
		CUDNN_TENSOR_NCHW,
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

	cudnnCheck(cudnnPoolingForward(cudnn,
		max2Des,
		alpha,
		conv2OutDes,
		d_conv2Out,
		beta,
		max2OutDes,
		d_max2Out));

	//--------------------------------------------------------conv3-------------------------------------------------------------------
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

	//load W3
	float* w3 = (float*)malloc(64 * 32 * 3 * 3 * sizeof(float));
	float* d_w3;
	cudaMalloc(&d_w3, 64 * 32 * 3 * 3 * sizeof(float));
	totalSpace += 64 * 32 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w3, w3, 64 * 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//(104, 104, 64)
	cudnnTensorDescriptor_t conv3OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv3OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv3OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		64,
		104,
		104));

	float* d_conv3Out;
	cudaMalloc(&d_conv3Out, 64 * 104 * 104  * sizeof(float));
	totalSpace += 64 * 104 * 104  * sizeof(float);


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

	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max2OutDes,
		w3Des,
		conv3Des,
		conv3OutDes,
		conv3Algo,
		&space));


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
	//don't forget to add the biases


	float b3[64] = { 0 };
	float* d_b3;
	cudaMalloc(&d_b3, 64 * sizeof(float));


	cudaMemcpy(d_b3, b3, 64 * sizeof(float), cudaMemcpyHostToDevice);

	add_biase << <dim3(835, 835), 1 >> >(d_conv3Out, d_b3, 104 * 104 * 64 , 104 * 104);
	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 3------------------------------------------------------------------
	////(104, 104, 64)
	leaky_relu << <dim3(835, 835), 1 >> > (d_conv3Out, .01, 104 * 104 * 64);

	//----------------------------------------------------max 3----------------------------------------------------------------
	//MaxPooling     2×2      2      (52, 52, 64)
	float* d_max3Out;
	cudaMalloc(&d_max3Out, 52 * 52 * 64 * sizeof(float));
	totalSpace += 52 * 52 * 64 * sizeof(float);
	cudnnTensorDescriptor_t max3OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max3OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max3OutDes,
		CUDNN_TENSOR_NCHW,
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

	//load W3
	float* w4 = (float*)malloc(128 * 64 * 3 * 3 * sizeof(float));
	float* d_w4;
	cudaMalloc(&d_w4, 128 * 64 * 3 * 3 * sizeof(float));
	totalSpace += 128 * 64 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w4, w4, 128 * 64 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//(52, 52, 128)
	cudnnTensorDescriptor_t conv4OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv4OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv4OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		128,
		52,
		52));

	float* d_conv4Out;
	cudaMalloc(&d_conv4Out, 128 * 52 * 52 * sizeof(float));
	totalSpace += 128 * 52 * 52 * sizeof(float);


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
	cudaFree(workSpace);
	cudaMalloc(&workSpace, space);
	totalSpace += space;
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

	float b4[128] = { 0 };
	float* d_b4;
	cudaMalloc(&d_b4, 128 * sizeof(float));


	cudaMemcpy(d_b4, b4, 128 * sizeof(float), cudaMemcpyHostToDevice);

	add_biase << <dim3(600, 600), 1 >> >(d_conv4Out, d_b4, 52 * 52 * 128 , 52 * 52);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 4------------------------------------------------------------------
	////(52, 52, 128)
	leaky_relu << <dim3(600, 600), 1 >> > (d_conv4Out, .01, 52 * 52 * 128);

	//----------------------------------------------------max 4----------------------------------------------------------------
	//MaxPooling     2×2      2      (26, 26, 128)
	float* d_max4Out;
	cudaMalloc(&d_max4Out, 26 * 26 * 128 * sizeof(float));
	totalSpace += 26 * 26 * 128 * sizeof(float);
	cudnnTensorDescriptor_t max4OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max4OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max4OutDes,
		CUDNN_TENSOR_NCHW,
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

	//load W3
	float* w5 = (float*)malloc(256 * 128 * 3 * 3 * sizeof(float));
	float* d_w5;
	cudaMalloc(&d_w5, 256 * 128 * 3 * 3 * sizeof(float));
	totalSpace += 256 * 128 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w5, w5, 256 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//(26, 26, 256)
	cudnnTensorDescriptor_t conv5OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv5OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv5OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		256,
		26,
		26));

	float* d_conv5Out;
	cudaMalloc(&d_conv5Out, 256 * 26 * 26 * sizeof(float));
	totalSpace += 256 * 26 * 26 * sizeof(float);


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

	float b5[256] = { 0 };
	float* d_b5;
	cudaMalloc(&d_b5, 256 * sizeof(float));


	cudaMemcpy(d_b5, b5, 256 * sizeof(float), cudaMemcpyHostToDevice);

	add_biase << <dim3(420, 420), 1 >> >(d_conv5Out, d_b5, 26 * 26 * 256 , 26 * 26);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 5------------------------------------------------------------------
	////(26, 26, 256)
	leaky_relu << <dim3 (420, 420), 1 >> > (d_conv5Out, .01, 26 * 26 * 256);

	//----------------------------------------------------max 5----------------------------------------------------------------
	//MaxPooling     2×2      2      (13, 13, 256)
	float* d_max5Out;
	cudaMalloc(&d_max5Out, 13 * 13 * 256 * sizeof(float));
	totalSpace += 13 * 13 * 256 * sizeof(float);
	cudnnTensorDescriptor_t max5OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max5OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max5OutDes,
		CUDNN_TENSOR_NCHW,
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

	//load W6
	float* w6 = (float*)malloc(512 * 256 * 3 * 3 * sizeof(float));
	float* d_w6;
	cudaMalloc(&d_w6, 512 * 256 * 3 * 3 * sizeof(float));
	totalSpace += 512 * 256 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w6, w6, 512 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//(13, 13, 512)
	cudnnTensorDescriptor_t conv6OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv6OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv6OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		512,
		13,
		13));

	float* d_conv6Out;
	cudaMalloc(&d_conv6Out, 512 * 13 * 13 * sizeof(float));
	totalSpace += 512 * 13 * 13 * sizeof(float);


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

	float b6[512] = { 0 };
	float* d_b6;
	cudaMalloc(&d_b6, 512 * sizeof(float));


	cudaMemcpy(d_b6, b6, 512 * sizeof(float), cudaMemcpyHostToDevice);

	add_biase << <dim3(300, 300), 1 >> > (d_conv6Out, d_b6, 13 * 13 * 512 , 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 6------------------------------------------------------------------
	////(13, 13, 512)
	leaky_relu << <dim3 (300, 300), 1 >> > (d_conv6Out, .01, 13 * 13 * 512);

	//----------------------------------------------------max 6----------------------------------------------------------------
	//MaxPooling     2×2      1      (13, 13, 512)
	//here there's padding and stride 1
	float* d_max6Out;
	cudaMalloc(&d_max6Out, 13 * 13 * 512 * sizeof(float));
	totalSpace += 13 * 13 * 512 * sizeof(float);
	cudnnTensorDescriptor_t max6OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max6OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max6OutDes,
		CUDNN_TENSOR_NCHW,
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
		1,
		1,
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

	//load W7
	float* w7 = (float*)malloc(1024 * 512 * 3 * 3 * sizeof(float));
	float* d_w7;
	cudaMalloc(&d_w7, 1024 * 512 * 3 * 3 * sizeof(float));
	totalSpace += 1024 * 512 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaMemcpy(d_w7, w7, 1024 * 512 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//(13 x  13 x 1024)
	cudnnTensorDescriptor_t conv7OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv7OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv7OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		1024,
		13,
		13));

	float* d_conv7Out;
	cudaMalloc(&d_conv7Out, 1024 * 13 * 13 * sizeof(float));
	totalSpace += 1024 * 13 * 13 * sizeof(float);


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

	float b7[1024] = { 0 };
	float* d_b7;
	cudaCheck(cudaMalloc(&d_b7, 1024 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b7, b7, 1024 * sizeof(float), cudaMemcpyHostToDevice));

	add_biase << <dim3(420, 420), 1 >> > (d_conv7Out, d_b7, 13 * 13 * 1024 , 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 7------------------------------------------------------------------
	////(13 x  13 x 1024)
	leaky_relu << <dim3 (420, 420), 1 >> > (d_conv7Out, .01, 13 * 13 * 1024);


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

	//load W8
	float* w8 = (float*)malloc(1024 * 1024 * 3 * 3 * sizeof(float));
	float* d_w8;
	cudaCheck(cudaMalloc(&d_w8, 1024 * 1024 * 3 * 3 * sizeof(float)));
	totalSpace += 1024 * 1024 * 3 * 3 * sizeof(float);
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w8, w8, 1024 * 1024 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
	//(13 x  13 x 1024)
	cudnnTensorDescriptor_t conv8OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv8OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv8OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		1024,
		13,
		13));

	float* d_conv8Out;
	cudaCheck(cudaMalloc(&d_conv8Out, 1024 * 13 * 13 * sizeof(float)));
	totalSpace += 1024 * 13 * 13 * sizeof(float);


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
	cout << "total space  " << totalSpace/(1024*1024) << "  MB\n";
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

	float b8[1024] = { 0 };
	float* d_b8;
	cudaCheck(cudaMalloc(&d_b8, 1024 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b8, b8, 1024 * sizeof(float), cudaMemcpyHostToDevice));

	add_biase << <dim3(420, 420), 1 >> > (d_conv8Out, d_b8, 13 * 13 * 1024 , 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution

	//-----------------------------------------------------relu 8------------------------------------------------------------------
	////(13 x  13 x 1024)
	leaky_relu << <dim3(420, 420), 1 >> > (d_conv8Out, .01, 13 * 13 * 1024);



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

	//load W9
	float* w9 = (float*)malloc(1024 * 125 * sizeof(float));
	float* d_w9;
	cudaCheck(cudaMalloc(&d_w9, 1024 * 125 * sizeof(float)));
	totalSpace += 1024 * 125  * sizeof(float);
	
	//copy weights to GPU
	cudaCheck(cudaMemcpy(d_w9, w9, 1024 * 125 * sizeof(float), cudaMemcpyHostToDevice));
	//(13 x  13 x 125)
	cudnnTensorDescriptor_t conv9OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv9OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv9OutDes,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		125,
		13,
		13));

	float* d_conv9Out;
	cudaCheck(cudaMalloc(&d_conv9Out, 125 * 13 * 13 * sizeof(float)));
	totalSpace += 125 * 13 * 13 * sizeof(float);
	cout << "total space " << totalSpace / (1024 * 1024) << " MB\n";

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

	float b9[125] = { 0 };
	float* d_b9;
	cudaCheck(cudaMalloc(&d_b9, 125 * sizeof(float)));


	cudaCheck(cudaMemcpy(d_b9, b9, 125 * sizeof(float), cudaMemcpyHostToDevice));

	add_biase << <dim3(150, 150), 1 >> > (d_conv9Out, d_b9, 13 * 13 * 125 , 13 * 13);

	//to be space effecient free workspace but make sure it doesn't include any data related to convolution
	long t2 = clock();
	cout <<"total space  "<< totalSpace / (1024 * 1024) << "	MB\n";
	cout << "time =  " << t2-t1<< "\n";

}
