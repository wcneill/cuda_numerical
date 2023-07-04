#include <vector>
#include <iostream>
#include "cu_integrate.cuh"
#include "cu_ops.cuh"


__host__
void checkLastErr() {
    auto res1 = cudaGetLastError();
    if (res1 != cudaSuccess) {
        std::cout << cudaGetErrorString(res1) << std::endl;
    }
}

__global__
void area_calc_kernel(int sections, float f[], float dx[], float result[]) {

    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < sections) {
        result[k] = dx[k] * (f[k] + f[k + 1]) / 2;
    }
}


/**
 * @brief Computes the area under a sampled curve.
 * 
 * @param n_sections The number of sections the function is divided into. Equal to #samples minus one. 
 * @param f function values sampled on a uniform grid
 * @param delta_x The spacing between samples. 
*/
__host__ 
float integrate_cuda(int n_sections, float f[], float delta_x[]) {

    int n_samples = n_sections + 1;

    // number of bytes to allocate for inputs and outputs of kernel
    size_t input_bytes = n_samples * sizeof(float);
    size_t result_bytes = n_sections * sizeof(float);

    // allocate memory for results on cpu
    float *cpu_results = (float*)malloc(result_bytes);

    // allocate memory for inputs and results on gpu
    float *gpu_samples; 
    float *gpu_results;
    float *gpu_dx;

    cudaMalloc(&gpu_dx, result_bytes);
    cudaMalloc(&gpu_samples, input_bytes);
    cudaMalloc(&gpu_results, result_bytes);
    checkLastErr();

    //copy cpu inputs to cuda array
    cudaMemcpy(gpu_samples, f, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dx, delta_x, result_bytes, cudaMemcpyHostToDevice);
    checkLastErr();

    // call kernel
    int num_threads = 256;
    int num_blocks = (n_sections + num_threads) / num_threads;
    area_calc_kernel<<<num_blocks, num_threads>>> (n_sections, gpu_samples, gpu_dx, gpu_results);
    checkLastErr();

    // copy data from gpu memory over to host (cpu) and free gpu memory
    cudaMemcpy(cpu_results, gpu_results, result_bytes, cudaMemcpyDeviceToHost);

    // do summation of section areas on the cpu
    float *partial_sums;
    cudaMalloc(&partial_sums, num_blocks * sizeof(float));
    sum_kernel<<<num_blocks, num_threads>>>(n_sections, gpu_results, partial_sums);
    sum_kernel<<<1, num_threads>>>(num_blocks, partial_sums, partial_sums);

    float area = 0.0f;
    cudaMemcpy(&area, partial_sums, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(partial_sums);
    cudaFree(gpu_results);
    free(cpu_results);
    return area;
}

