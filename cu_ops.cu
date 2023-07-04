#include "cu_ops.cuh"
#include <assert.h>
#include <iostream>

#define SIZE 256

void printLastCudaError() { 
    auto res = cudaGetLastError();
    if (res != cudaSuccess) {
        std::cout << cudaGetErrorString(res) << std::endl;
    }
}

__global__
void sum_kernel(int n, float x[], float result[]) {
    __shared__ float partial_sum[SIZE * sizeof(float)];

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < n) {
        partial_sum[threadIdx.x] = x[thread_idx];
    }
    __syncthreads();

    // this is naive as it leaves a lot of threads inactive after the first iteration
    for (int stride = 1; stride < blockDim.x; stride*=2) {
        int index = threadIdx.x * stride * 2;
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index] + partial_sum[index + stride];
        }
        __syncthreads();
    }

    // each thread block is responsible for computing one partial sum
    // so we store the result according to block index. 
    if (threadIdx.x == 0) { 
        result[blockIdx.x] = partial_sum[0];
    }

}

__global__ 
void mult_kernel(int n, float x[], float scalar, float result[]) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        result[tid] = scalar * x[tid];
    }
}

__global__
void mult_kernel(int n, float x1[], float x2[], float result[]) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        result[tid] = x1[tid] * x2[tid];
    }
}

__host__ 
std::vector<float> mult(std::vector<float> x, float scalar) {

    int n_inputs = (int)x.size();

    int n_threads = 256;
    int n_blocks = (n_inputs + n_threads) / n_threads;
    size_t bytes = n_inputs * sizeof(float);

    std::vector<float> y(n_inputs);

    float *d_x; // gpu "device" inputs
    float *d_y; // gpu "device" outputs

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
    printLastCudaError();

    mult_kernel<<<n_blocks, n_threads>>>(n_inputs, d_x, scalar, d_y);
    printLastCudaError();

    cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
    printLastCudaError();

    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}


__host__ 
std::vector<float> mult(std::vector<float> x1, std::vector<float> x2) {

    assert (x1.size() == x2.size());
    int n_inputs = (int)x1.size();

    int n_threads = 256;
    int n_blocks = (n_inputs + n_threads) / n_threads;
    size_t bytes = n_inputs * sizeof(float);

    std::vector<float> y(n_inputs);

    float *d_x1; // gpu "device" inputs
    float *d_x2;
    float *d_y; // gpu "device" outputs
    
    cudaMalloc(&d_x1, bytes);
    cudaMalloc(&d_x2, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x1, x1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2.data(), bytes, cudaMemcpyHostToDevice);
    mult_kernel<<<n_blocks, n_threads>>>(n_inputs, d_x1, d_x2, d_y);
    cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);

    return y;
}

__host__
float sum(std::vector<float> x) {

    int block_size = SIZE;
    int grid_size = (x.size() + block_size) / block_size;

    // Probably inefficient to allocate new memory and copy inputs, but this is a workaround 
    // for the fact that sum reductions are computed based on thread block size, which may 
    // not be divisible by the number of input elements. 
    int remainder = x.size() % block_size; 
    int padded_size = x.size() + remainder;
    float *h_inputs = (float*) calloc(padded_size, sizeof(float));

    float *d_inputs; 
    float *d_sums;
    cudaMalloc(&d_sums, grid_size * sizeof(float));
    cudaMalloc(&d_inputs, padded_size * sizeof(float));

    cudaMemcpy(d_inputs, h_inputs, padded_size * sizeof(float), cudaMemcpyHostToDevice);

    sum_kernel<<<grid_size, block_size>>>(padded_size, d_inputs, d_sums);
    sum_kernel<<<1, block_size>>>(block_size, d_sums, d_sums); 

    float result = 0;
    result += d_sums[0];

    cudaFree(d_sums);
    cudaFree(d_inputs);
    free(h_inputs);

    return result;
}
