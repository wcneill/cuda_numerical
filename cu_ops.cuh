#include <cuda_runtime.h>
#include <vector>


__global__ 
void mult(int n, float x[], float scalar, float result[]);

__global__ 
void mult(int n, float x1[], float x2[], float result[]);

__host__ 
std::vector<float> mult(std::vector<float> x, float scalar);

__host__ 
std::vector<float> mult(std::vector<float> x1, std::vector<float> x2);