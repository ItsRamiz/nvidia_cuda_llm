#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        // Very simple math
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    vectorAdd<<<n, 1>>>(d_a, d_b, d_c, n);  // ❌ horrible launch config

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Result[0] = " << h_c[0] << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
