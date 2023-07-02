#include <cuda_runtime.h>

__global__ 
void area_calc_kernel(int n, float f[], float dx[], float result[]);

__host__ 
float integrate_cuda_cpu(int n_samples, float f[], float delta_x[]);

__host__
void checkLastErr();