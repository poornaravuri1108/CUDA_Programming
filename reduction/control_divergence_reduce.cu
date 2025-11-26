#include <cuda_runtime.h>
#include <iostream>

__global__ void control_divergence_reduce(float* d_input, float* d_output){
	int i = threadIdx.x;
	for(int stride=blockDim.x; stride>=1; stride /= 2){
		if(threadIdx.x<stride){
			d_input[i]+=d_input[i+stride];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		*d_output = d_input[0];
	}
}

int main(){

	const int size=2048;
	const int bytes = size*sizeof(float);

	float* h_input = new float[size];
	float* h_output = new float;

	for(int i=0; i<size; i++){
		h_input[i]=1.0f;
	}

	float* d_input;
	float* d_output;

	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_output, sizeof(float));

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
	control_divergence_reduce<<<1, size/2>>>(d_input, d_output);
	cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Sum is: " << *h_output << std::endl;

	delete[] h_input;
	delete h_output;
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}