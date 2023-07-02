// #include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "integrate.h"
#include "cu_ops.cuh"
#include "cu_integrate.cuh"


float mult(float x, float a) {
    return x * a;
}


int main() {

    float a = 0.0f;
    float b = 2.0f;
    int N = 10000; // number of "panel" sections
    float delta_x = (b - a) / N; 

    std::vector<float> dx(N, delta_x);
    std::vector<float> x(N + 1);
    std::vector<float> f(N + 1);
    init_vector(a, b, N, x);    
    
    for (int i = 0; i < N + 1; i++){
        f[i] = mult(x[i], x[i]);
    }

    float area1 = integrate_cpu(N, f.data(), dx.data());
    std::cout << "CPU: The area is " << area1 << std::endl; 

    // test cuda vector mult
    std::vector<float> y = mult(x, x);

    // integrate on gpu
    float area2 = integrate_cuda_cpu(N, y.data(), dx.data());
    std::cout << "GPU: The area is " << area2 << "\n" << std::endl;

}