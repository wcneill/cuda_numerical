#include <iostream>
#include <vector>
#include "integrate.h"

float integrate_cpu(int n, float f[], float dx[]){

    float temp_sum = 0;
    for (int k = 0; k < n; k++) {
        float area = dx[k] * (f[k] + f[k + 1]) / 2;
        temp_sum += area;
    }

    return temp_sum;
}

float init_vector(float a, float b, int num_sections, std::vector<float> &v) {
    float deltax = (b - a) / num_sections;
    for (int i = 0; i <= num_sections; i++) {
        v[i] = a + i * deltax;
    }
    return deltax;
}