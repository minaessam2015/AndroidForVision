
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

#define TEST 1
using namespace std;


__global__ void flatten(float* input, float* out, int batch, int n, int h, int w) {
	int cond1 = (((threadIdx.y*w) + threadIdx.x));
	int cond2 = (threadIdx.x*n) + blockIdx.x + threadIdx.y*w*n;
	if (cond2<(n*h*w)) {
		printf("index  %d    %d    \n", ((threadIdx.y*w) + threadIdx.x) + (blockIdx.x*w*h), cond2);
		//printf("%f\n", input[((threadIdx.y*w) + threadIdx.x) ]);
		out[cond1 + (blockIdx.x*w*h)] = input[cond2];
	}
}

__global__ void addBias(float* vector,float bias,int size) {
	int index = blockIdx.y*blockIdx.x + blockIdx.x;
	if (index < size) {
		//printf(" from addBias  %f \n", vector[index]);
		vector[index] += bias;
		//printf(" from addBias  %f \n", vector[index]);
	}
}

void readWeights(float weights[][3][3][3], int m/*output*/, int n/*input*/, int h, int w, string baseFileName) {

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			string fileName = "weights/"+baseFileName + std::to_string(j) + "X" + std::to_string(i) + ".txt";
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

void readWeights(float* weights, int size, string baseFileName) {
	ifstream in("weights/" + baseFileName, std::ifstream::in);
	//cout << baseFileName << "\n";
	if (!in.is_open())
	
	{
		cout << "file didn't open \n";
		return;
	}
	char c;

	string s = "";
	for (int i = 0; i < size; i++) {


		while (in.get(c)) {
			if (c == ' '&&s.length() == 0)continue;
			if (c != ' '&&c != '\n')s += c;
			else
			{
				break;
			}

		}
		if (s.length()>0) weights[i] = std::stof(s);
		//cout << i<<"  "<<s << "\n";
		s = "";
	}
	in.close();

}



void readWeights(float weights[][8][3][3], int m/*output*/, int n/*input*/, int h, int w, string baseFileName) {
	//file will be in format baseFileName nXm .txt
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

							//cout << "breaking with c " << c << "\n";
							break;
						}
						//cout << c << " ";
					}
					if (s.length()>0)weights[i][j][k][l] = std::stof(s);
					s = "";
					//cout << std::stof(s) << "\n";
				}
			}
		}
	}

}


cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (image.empty()) { cerr << "couldn't open image\n"; }
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void save_image(const char* output_filename,
	float* buffer,
	int height,
	int width) {
	cv::Mat output_image(height, width, CV_32FC3, buffer);
	// Make negative values zero.
	cv::threshold(output_image,
		output_image,
		/*threshold=*/0,
		/*maxval=*/0,
		cv::THRESH_TOZERO);
	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

#define cudnnCheck(exp)																\
{																					\
cudnnStatus_t status=(exp);															\
if(status!=CUDNN_STATUS_SUCCESS){													\
cerr<<"Error at line "<<__LINE__<<cudnnGetErrorString(status)<<"\n";				\
std::exit(EXIT_FAILURE);															\
}																					\
																					\
} 

int main() {


	float* alpha=new float;
	alpha[0] = 1.0;
	float* beta=new float;
	beta[0] = 0.0;


	char* imageName = "car2.png";
	cout << imageName << "\n";
	cv::Mat image = load_image(imageName);
	//for (int i = 0; i < 5; i++) {
	//	cout<<(image.at<float>(i, 0))<<" ";
	//}
	//cout << "\n";
	cudnnHandle_t cudnn;
	cudnnCheck(cudnnCreate(&cudnn));
	cout << "image dims " << image.rows << " X " << image.cols << "\n";
	//input image
	long long t1 = clock();
	cudnnTensorDescriptor_t inputImageDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&inputImageDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(inputImageDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		3,
		32,
		32
	));
//-----------------------------------------------------------------CONV1------------------------------------------------------------
	//W1
	cudnnFilterDescriptor_t conv1W;
	cudnnCheck(cudnnCreateFilterDescriptor(&conv1W));
	cudnnCheck(cudnnSetFilter4dDescriptor(conv1W,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		8,
		3,
		3,
		3));
	cudnnTensorDescriptor_t conv1Out;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv1Out));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv1Out,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		8,
		32,
		32));

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

	cudnnConvolutionFwdAlgo_t conv1AlgDes;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		inputImageDes,
		conv1W,
		conv1Des,
		conv1Out,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv1AlgDes));

	size_t workspace_bytes = 0;


	
	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		inputImageDes,
		conv1W,
		conv1Des,
		conv1Out,
		conv1AlgDes,
		&workspace_bytes));
	//cout << "required space for conv1 " << workspace_bytes / (1024) << "\n";
	
	void* d_workspace = nullptr;
	cudaMalloc(&d_workspace, workspace_bytes);
	//image alloc 
	float* d_image;
	int imageSize = 3 * image.rows*image.cols;
	cudaMalloc(&d_image, imageSize * sizeof(float));
	cudaMemcpy(d_image, image.ptr<float>(0), imageSize*sizeof(float), cudaMemcpyHostToDevice);



	//output from conv1 [ 1 * 8 * 32 * 32 ]
	float* d_conv1Out;
	int conv1OutSize = 1 * 8 * 32 * 32;
	cudaMalloc(&d_conv1Out, conv1OutSize * sizeof(float));
	
	//conv1 kernel [3 * 3 * 3 * 8]
	float* d_conv1W;
	int conv1WSize = 3 * 3 * 3 * 8;
	cudaMalloc(&d_conv1W, conv1WSize * sizeof(float));
	//TODO
	//get the data 
	//copy the data to the GPU
	float h_conv1W[8][3][3][3];
	readWeights(h_conv1W, 8, 3, 3, 3,"conv1Weights");
	cudaMemcpy(d_conv1W, h_conv1W, sizeof(h_conv1W), cudaMemcpyHostToDevice);
	//cout << "conv1 weights\n";
	//
	////test for conv1 weights
	//float h_conv1WTest[20];
	//cudaMemcpy(h_conv1WTest, d_conv1W, sizeof(h_conv1WTest), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 20; i++) {
	//	cout << h_conv1WTest[i] << " ";
	//}
	//cout << "\n";
	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		inputImageDes,
		d_image,
		conv1W,
		d_conv1W,
		conv1Des,
		conv1AlgDes,
		d_workspace,
		workspace_bytes,
		beta,
		conv1Out,
		d_conv1Out));

	////test relu 1 out 
	//float h_conv1Test[20];
	//cudaMemcpy(h_conv1Test, d_conv1Out, sizeof(h_conv1Test), cudaMemcpyDeviceToHost);
	//cout << "conv1 20 values\n";
	//for (int i = 0; i < 20; i++) {
	//	cout << h_conv1Test[i] << " ";
	//}
	//cout << "\n\n";

//------------------------------------------------------------RELU 1--------------------------------------------------------

	//relu1 in=[1 *32 *32 *8] out the same
	cudnnActivationDescriptor_t relu1Des;
	cudnnCheck(cudnnCreateActivationDescriptor(&relu1Des));
	cudnnCheck(cudnnSetActivationDescriptor(relu1Des,
		CUDNN_ACTIVATION_RELU,
		CUDNN_NOT_PROPAGATE_NAN,
		0.0));

	//// INPUT data is same from conv 1 
	////allocate output data [1 *32 *32 *8] 


	//cudnnTensorDescriptor_t relu1InputDes;
	//cudnnCheck(cudnnCreateTensorDescriptor(&relu1InputDes));
	//cudnnCheck(cudnnSetTensor4dDescriptor(relu1InputDes,
	//	CUDNN_TENSOR_NHWC,
	//	CUDNN_DATA_FLOAT,
	//	1,
	//	8,
	//	32,
	//	32));

	cudnnTensorDescriptor_t relu1OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&relu1OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(relu1OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		8,
		32,
		32));
	float* d_relu1Out;
	int relu1Size = 1 * 32 * 32 * 8;
	cudaMalloc(&d_relu1Out, relu1Size * sizeof(float));


	
	cudnnCheck(cudnnActivationForward(cudnn,
		relu1Des,
		alpha,
		conv1Out,
		d_conv1Out,
		beta,
		relu1OutDes,
		d_relu1Out
		));
	
	////test relu 1 out 
	//float h_relu1Test[20];
	//cudaMemcpy(h_relu1Test, d_relu1Out, sizeof(h_relu1Test), cudaMemcpyDeviceToHost);
	//cout << "relu1 20 values\n";
	//for (int i = 0; i < 20; i++) {
	//	cout << h_relu1Test[i] << " ";
	//}
	//cout << "\n\n";
	//----------------------------------------MAX 1 pooling ---------------------------------
	//MAX polling layer 
	//in d_relu1Out
	//out [1 * 4 * 4 * 8]

	cudnnPoolingDescriptor_t max1Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max1Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max1Des,
		CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN,
		8,
		8,
		0,
		0,
		8,
		8));

	cudnnTensorDescriptor_t max1OutputDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max1OutputDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max1OutputDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		8,
		4,
		4));

	float* d_max1Out;
	int max1Size = 1 * 4 * 4 * 8;
	cudaMalloc(&d_max1Out, max1Size* sizeof(float));
	
	cudnnCheck(cudnnPoolingForward(cudnn,
		max1Des,
		alpha,
		relu1OutDes,
		d_relu1Out,
		beta,
		max1OutputDes,
		d_max1Out
		));

	////Test
	//float h_max1Test[20];
	//cudaMemcpy(h_max1Test, d_max1Out, sizeof(h_max1Test), cudaMemcpyDeviceToHost);
	//cout << "max1 20 values\n";
	//for (int i = 0; i < 20; i++) {
	//	cout << h_max1Test[i] << " ";
	//}
	//cout << "\n\n";

	//-------------------------------------------conv2 layer---------------------------------------------

	//cudnnTensorDescriptor_t conv2InputDes;
	//cudnnCheck(cudnnCreateTensorDescriptor(&conv2InputDes));
	//cudnnCheck(cudnnSetTensor4dDescriptor(conv2InputDes,
	//	CUDNN_TENSOR_NHWC,
	//	CUDNN_DATA_FLOAT,
	//	1,
	//	8,
	//	4,
	//	4
	//));
	//W1
	cudnnFilterDescriptor_t conv2W;
	cudnnCheck(cudnnCreateFilterDescriptor(&conv2W));
	cudnnCheck(cudnnSetFilter4dDescriptor(conv2W,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		16,
		8,
		3,
		3));
	cudnnTensorDescriptor_t conv2Out;
	cudnnCheck(cudnnCreateTensorDescriptor(&conv2Out));
	cudnnCheck(cudnnSetTensor4dDescriptor(conv2Out,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		16,
		4,
		4));

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
	cudnnConvolutionFwdAlgo_t conv2AlgDes;
	cudnnCheck(cudnnGetConvolutionForwardAlgorithm(cudnn,
		max1OutputDes,
		conv2W,
		conv2Des,
		conv2Out,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&conv2AlgDes));

	size_t workspace_bytes2 = 0;
	cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		max1OutputDes,
		conv2W,
		conv2Des,
		conv2Out,
		conv2AlgDes,
		&workspace_bytes2));
	//cout << "space for conv2 " << workspace_bytes2 / 1024 << "\n";
	void* d_workspace2 = nullptr;
	cudaMalloc(&d_workspace2, workspace_bytes2);


	//output from conv2 [ 1 * 4 * 4 * 16 ]
	float* d_conv2Out;
	int conv2OutSize = 1 * 4 * 4 * 16;
	cudaMalloc(&d_conv2Out, conv2OutSize*sizeof(float));
	

	//conv2 kernel [3 * 3 * 8 * 16]
	float* d_conv2W;
	int conv2WSize = 3 * 3 * 8 * 16;
	cudaMalloc(&d_conv2W, conv2WSize*sizeof(float));
	//TODO
	//get the data W2
	//copy the data to the GPU

	float h_conv2W[16][8][3][3];
	readWeights(h_conv2W, 16, 8, 3, 3, "conv2Weights");
	cudaMemcpy(d_conv2W, h_conv2W, sizeof(h_conv2W), cudaMemcpyHostToDevice);

	cudnnCheck(cudnnConvolutionForward(cudnn,
		alpha,
		max1OutputDes,
		d_max1Out,
		conv2W,
		d_conv2W,
		conv2Des,
		conv2AlgDes,
		d_workspace2,
		workspace_bytes2,
		beta,
		conv2Out,
		d_conv2Out));

	////Test
	//float h_conv2Test[20];
	//cudaMemcpy(h_conv2Test, d_conv2Out, sizeof(h_conv2Test), cudaMemcpyDeviceToHost);
	//cout << "conv2 20 values\n";
	//for (int i = 0; i < 20; i++) {
	//	cout << h_conv2Test[i] << " ";
	//}
	//cout << "\n\n";

//-------------------------------------------------------------RELU 2--------------------------------------------------------

	//relu2 in=[1 *4 *4 *8] out the same
	cudnnActivationDescriptor_t relu2Des;
	cudnnCheck(cudnnCreateActivationDescriptor(&relu2Des));
	cudnnCheck(cudnnSetActivationDescriptor(relu2Des,
		CUDNN_ACTIVATION_RELU,
		CUDNN_NOT_PROPAGATE_NAN,
		0.0));

	// INPUT data is same from conv 1 
	//allocate output data [ 1 *4 *4 *16 ] 


	//cudnnTensorDescriptor_t relu1InputDes;
	//cudnnCheck(cudnnCreateTensorDescriptor(&relu1InputDes));
	//cudnnCheck(cudnnSetTensor4dDescriptor(relu1InputDes,
	//	CUDNN_TENSOR_NHWC,
	//	CUDNN_DATA_FLOAT,
	//	1,
	//	8,
	//	32,
	//	32));

	cudnnTensorDescriptor_t relu2OutDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&relu2OutDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(relu2OutDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		16,
		4,
		4));

	float* d_relu2Out;
	int relu2Size = 1 * 4 * 4 * 16;
	cudaMalloc(&d_relu2Out, relu2Size*sizeof(float));


	cudnnCheck(cudnnActivationForward(cudnn,
		relu2Des,
		alpha,
		conv2Out,
		d_conv2Out,
		beta,
		relu2OutDes,
		d_relu2Out
	));


	////Test
	//float h_relu2Test[4*4*16];
	//cudaMemcpy(h_relu2Test, d_relu2Out, sizeof(h_relu2Test), cudaMemcpyDeviceToHost);
	//cout << "relu2 16*16 values\n";
	//for (int i = 0; i < 4*4*16; i++) {
	//	cout << h_relu2Test[i] << " ";
	//}
	//cout << "\n\n";

//------------------------------------------------------MAX 2 pooling ----------------------------------------
	//MAX polling layer 
	//in d_relu2Out
	//out [ 1 * 1 * 1 * 16 ]

	cudnnPoolingDescriptor_t max2Des;
	cudnnCheck(cudnnCreatePoolingDescriptor(&max2Des));
	cudnnCheck(cudnnSetPooling2dDescriptor(max2Des,
		CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN,
		4,
		4,
		0,
		0,
		4,
		4));

	cudnnTensorDescriptor_t max2OutputDes;
	cudnnCheck(cudnnCreateTensorDescriptor(&max2OutputDes));
	cudnnCheck(cudnnSetTensor4dDescriptor(max2OutputDes,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		16,
		1,
		1));

	float* d_max2Out;
	int max2Size = 1 * 1 * 1 * 16;
	cudaMalloc(&d_max2Out, max2Size*sizeof(float));

	cudnnCheck(cudnnPoolingForward(cudnn,
		max2Des,
		alpha,
		relu2OutDes,
		d_relu2Out,
		beta,
		max2OutputDes,
		d_max2Out
	));

	//float* d_flatOut;
	//cudaMalloc(&d_flatOut,max2Size*sizeof(float));

	//cudaError_t error;
	//flatten<<<>>>(d_max2Out, d_flatOut, 1, 16, 1, 1);
	//error=cudaDeviceSynchronize();
	//if (error != cudaSuccess) cout << "error flatten\n";



	////Test
	//float h_max2Test[16];
	//cudaMemcpy(h_max2Test, d_max2Out, sizeof(h_max2Test), cudaMemcpyDeviceToHost);
	//cout << "max2 16 values\n";
	//for (int i = 0; i <  16; i++) {
	//	cout << h_max2Test[i] << " ";
	//}
	//cout << "\n\n";

//----------------------------------------------------Fully connected-----------------------------------------------------------

	cublasHandle_t cublas;
	cublasCreate(&cublas);

	//create the fully connected weights & copy to GPU

	float h_fullyWeights[16];
	readWeights(h_fullyWeights, 16, "fc1Weights.txt");
	float* d_fullyWeights;
	float bias = 0;
	readWeights(&bias, 1, "fc1Bias.txt");
	cudaMalloc(&d_fullyWeights, sizeof(h_fullyWeights));
	cudaMemcpy(d_fullyWeights, h_fullyWeights, sizeof(h_fullyWeights), cudaMemcpyHostToDevice);

	//float h_max2Out[16];
	//cudaMemcpy(h_max2Out, d_max2Out, sizeof(h_fullyWeights), cudaMemcpyDeviceToHost);
	//cout << "fully connected weights\n";
	//for (int i = 0; i < 16; i++)cout << h_fullyWeights[i] << " ";
	//cout << "\n\n";

	float* d_tmp;
	cudaMalloc(&d_tmp,sizeof(float));

	//float fullyBias=0;
	//cudaMemcpy(d_tmp, &fullyBias, sizeof(float), cudaMemcpyHostToDevice);

	cublasStatus_t s= cublasSgemm_v2(cublas,
		CUBLAS_OP_N, CUBLAS_OP_N,
		1, 1, 16,
		alpha,
		d_max2Out,  1,
		d_fullyWeights, 16,
		beta,
		d_tmp,  1);
	if (s != CUBLAS_STATUS_SUCCESS) {
		std::cout << "error cublas\n";
	}
	addBias <<<dim3(1,1), dim3(1) >> > (d_tmp,bias,1);

	cudnnActivationDescriptor_t activationDes;
	cudnnCheck(cudnnCreateActivationDescriptor(&activationDes ));
	cudnnCheck(cudnnSetActivationDescriptor(activationDes,
		CUDNN_ACTIVATION_SIGMOID,
		CUDNN_PROPAGATE_NAN,
		0));
	cudnnTensorDescriptor_t inputToFully;
	cudnnCheck(cudnnCreateTensorDescriptor(&inputToFully));
	cudnnCheck(cudnnSetTensor4dDescriptor(inputToFully,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1,
		1,
		1,
		1));

	cudnnCheck(cudnnActivationForward(cudnn,
		activationDes,
		alpha,
		inputToFully,
		d_tmp,
		beta,
		inputToFully,
		d_tmp));
	cout << "total time elapsed " <<clock()-t1 << "\n";
	float result=3;
	cudaMemcpy(&result, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
	cout << result << "\n";
	
	cudnnCheck(cudnnDestroyTensorDescriptor(inputImageDes));
	cudnnCheck(cudnnDestroyTensorDescriptor(conv1Out));
	cudnnCheck(cudnnDestroyConvolutionDescriptor(conv1Des));
	cudnnCheck(cudnnDestroyFilterDescriptor(conv1W));
	//continue


}