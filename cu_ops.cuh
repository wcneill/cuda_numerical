#include <cuda_runtime.h>
#include <vector>


// sum reduction --------
__global__ 
void sum_kernel(int n, float x[], float result[]);
__host__ 
float sum(std::vector<float> x);

// scalar multiply ---------
__global__ 
void mult_kernel(int n, float x[], float scalar, float result[]);
__host__ 
std::vector<float> mult(std::vector<float> x, float scalar);

// vector multiply --------
__global__ 
void mult_kernel(int n, float x1[], float x2[], float result[]);
__host__ 
std::vector<float> mult(std::vector<float> x1, std::vector<float> x2);