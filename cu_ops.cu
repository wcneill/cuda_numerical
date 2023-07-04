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

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < n) {
        // load shared memory
        partial_sum[threadIdx.x] = x[global_idx];
    }
    __syncthreads();


    for (int stride = 1; stride < blockDim.x; stride*=2) {

        int shmem_index = threadIdx.x * stride * 2;
        int input_index = shmem_index + blockDim.x * blockIdx.x;

        // check bounds of shared memory and bournds of input!
        bool is_in_bounds = (shmem_index < blockDim.x) 
                            & (shmem_index + stride < blockDim.x)
                            & (input_index < n)
                            & (input_index + stride < n);

        if (is_in_bounds) {
            partial_sum[shmem_index] += partial_sum[shmem_index + stride];
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

    int n_threads = SIZE;
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

    int n_threads = SIZE;
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
    size_t grid_size = (x.size() + block_size) / block_size;

    float *h_result = (float*) malloc(sizeof(float));

    //device side stuff;
    float *d_inputs; 
    float *d_sums;
    cudaMalloc(&d_sums, grid_size * sizeof(float));
    cudaMalloc(&d_inputs, x.size() * sizeof(float));
    cudaMemcpy(d_inputs, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    printLastCudaError();

    // launch kernel 2x for complete reduction
    sum_kernel<<<grid_size, block_size>>>(x.size(), d_inputs, d_sums);
    printLastCudaError();

    sum_kernel<<<1, block_size>>>(grid_size, d_sums, d_sums);
    printLastCudaError(); 

    cudaMemcpy(h_result, d_sums, sizeof(float), cudaMemcpyDeviceToHost);
    printLastCudaError();

    float result = 0;
    result += h_result[0];

    cudaFree(d_sums);
    cudaFree(d_inputs);
    free(h_result);

    return result;
}
