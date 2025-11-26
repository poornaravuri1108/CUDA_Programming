#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void multiblock_reduction(float* d_input, float* d_output, int size){
    __shared__ float input_s[BLOCK_DIM];

    int segment = 2 * blockDim.x * blockIdx.x;
    int t = threadIdx.x;

    int i = segment + t;

    float a = (i < size) ? d_input[i] : 0.0f;
    float b = (i + blockDim.x < size) ? d_input[i + blockDim.x] : 0.0f;

    input_s[t] = a + b;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(d_output, input_s[0]);
    }
}




int main() {
    int size = 100000;
	int bytes = size * sizeof(float);

	float* h_input = new float[size];
	float h_output = 0.0f;

	for(int i = 0; i < size; i++){
	    h_input[i] = 1.0f;
	}

	float *d_input, *d_output;
	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_output, sizeof(float));

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	cudaMemset(d_output, 0, sizeof(float));

	int numBlocks = (size + 2*BLOCK_DIM - 1) / (2*BLOCK_DIM);

	multiblock_reduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);
	cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Sum is: " << h_output << std::endl;

	cudaFree(d_input);
	cudaFree(d_output);

	delete[] h_input;

}

/*
#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void SharedMemoryReduction(float* input, float* output, int n) {
    __shared__ float input_s[BLOCK_DIM]; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int t = threadIdx.x; 

    if (idx < n) {
        input_s[t] = input[idx];
    } else {
        input_s[t] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride && idx + stride < n) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}



int main() {
    const int size = 100000;
    const int bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float;

    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; 
    }

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    float zero = 0.0f;
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    SharedMemoryReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum is " << *h_output << std::endl;

    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
*/