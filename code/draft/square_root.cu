#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void sqrtKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Unnecessarily repeated expensive math
        for (int j = 0; j < 1000; j++) {
            data[i] = sqrt(data[i]);
        }
    }
}

int main() {
    int n = 1 << 18;
    size_t size = n * sizeof(float);

    float* h_data = new float[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = 1024.0f;
    }

    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    sqrtKernel<<<n, 1>>>(d_data, n);  // ❌ 1 thread per block again

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    std::cout << "Result[0] = " << h_data[0] << std::endl;

    delete[] h_data;
    cudaFree(d_data);
}
