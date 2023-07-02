#include "cu_ops.cuh"
#include <assert.h>
#include <iostream>

void printLastCudaError() { 
    auto res = cudaGetLastError();
    if (res != cudaSuccess) {
        std::cout << cudaGetErrorString(res) << std::endl;
    }
}

__global__ 
void mult(int n, float x[], float scalar, float result[]) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        result[tid] = scalar * x[tid];
    }
}

__global__
void mult(int n, float x1[], float x2[], float result[]) {
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

    mult<<<n_blocks, n_threads>>>(n_inputs, d_x, scalar, d_y);
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
    mult<<<n_blocks, n_threads>>>(n_inputs, d_x1, d_x2, d_y);
    cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);

    return y;
}
