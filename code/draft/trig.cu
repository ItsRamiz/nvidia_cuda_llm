#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void trigKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        data[i] = sin(data[i]) + cos(data[i]) + tan(data[i]);
    }
}

int main() {
    int n = 1 << 17;
    size_t size = n * sizeof(float);

    float* h = new float[n];
    for (int i = 0; i < n; i++) {
        h[i] = 0.5f;
    }

    float* d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    trigKernel<<<n, 1>>>(d, n);  // ❌ again terrible launch

    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    std::cout << h[0] << std::endl;

    delete[] h;
    cudaFree(d);
}
