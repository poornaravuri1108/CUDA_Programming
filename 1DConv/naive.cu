#include "cuda_runtime.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>

__global__ void conv_1d_kernel(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, int input_size, int kernel_size){
	int tid = threadIdx.x+blockDim.x*blockIdx.x;
	int output_size = input_size-kernel_size+1;
	if(tid>=output_size) return;

	float tmp=0.0f;
	for(int j=0; j<kernel_size; ++j){
		tmp+=input[tid+j]*kernel[j];
	}
	output[tid]=tmp;
}

extern 'c' void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size){
	int output_size = input_size-kernel_size+1;
	int threadsPerBlock = 256;
	int blocksPerGrid = (output_size+threadsPerBlock-1)/threadsPerBlock;

	conv_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
	cudaDeviceSynchronize();
}

void convolution_1d_cpu(const float* input, const float* kernel, float* output, int input_size, int kernel_size){
	int output_size = input_size-kernel_size+1;
	for(int i=0; i<output_size; ++i){
		float tmp = 0.0f;
		for(int j=0; j<kernel_size; ++j){
			tmp += input[i+j]*kernel[j];
		}
		output[i]=tmp;
	}	
}


int main(){
	int input_size = 1<<16;
	int kernel_size = 15;
	int iters = 10;

	int output_size = input_size-kernel_size+1;
	std::vector<float> h_input(input_size);
	std::vector<float> h_kernel(kernel_size);
	std::vector<float> h_output_cpu(output_size);
	std::vector<float> h_output_gpu(output_size);

	for(int i=0; i<input_size; ++i){
		h_input[i] = std::sin(0.001f*i);
	}

	for(int i=0; i<kernel_size; ++i){
		h_kernel[i] = 1.0f/kernel_size;
	}

	float *d_input = nullptr;
	float *d_kernel = nullptr;
	float *d_output = nullptr;

	cudaMalloc(&d_input, input_size*sizeof(float));
	cudaMalloc(&d_kernel, kernel_size*sizeof(float));
	cudaMalloc(&d_output, output_size*sizeof(float));

	cudaMemcpy(d_input, h_input.data(), input_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, h_kernel.data(), kernel_size*sizeof(float), cudaMemcpyHostToDevice);

	//warmup
	solve(d_input, d_kernel, d_output, input_size, kernel_size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for(int it=0; it<iters; ++it){
		solve(d_input, d_kernel, d_output, input_size, kernel_size);
	}
	cudaEventRecord(&stop);
	cudaEventElapsedTime(&ms, start, stop);
	float time_per_iter = ms/iters;

	cudaMemcpy(h_output_gpu.data(), d_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);

	convolution_1d_cpu(h_input.data(), h_kernel.data(),
                       h_output_cpu.data(), input_size, kernel_size);

    // Compare GPU vs CPU
    double max_abs_err = 0.0;
    for (int i = 0; i < output_size; ++i) {
        double diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > max_abs_err) max_abs_err = diff;
    }
    std::cout << "Max abs error (GPU vs CPU): " << max_abs_err << "\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_kernel);
	cudaFree(d_output);


}