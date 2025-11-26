#include <cuda_runtime.h>
#include <iostream>

__global__ void simple_reduce(float* d_input, float* d_output){
	unsigned int i = 2*threadIdx.x;
	for(unsigned int stride=1; stride <= blockDim.x; stride *= 2){
		if(threadIdx.x%stride == 0){
			d_input[i] += d_input[i+stride];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		*d_output = d_input[0];
	}
}

int main(){

	//input size
	int size=2048;
	int bytes = size*sizeof(float);

	float* h_input = new float[size];
	float* h_output = new float;

	float* d_input;
	float* d_output;

	for(int i=0; i<size; i++){
		h_input[i] = 1.0f;
	}

	//device memory allocation
	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_output, sizeof(float));

	//data transfer from host to device
	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	//reduction kernel
	simple_reduce<<<1, size/2>>>(d_input, d_output);

	//data transfer from device to host
	cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Sum is: " << *h_output << std::endl;

	delete[] h_input;
	delete h_output;
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}