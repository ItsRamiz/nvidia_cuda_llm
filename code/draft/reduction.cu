#include <cuda_runtime.h>
#include <iostream>

__global__ void sumKernel(float* data, float* result, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += data[i];   // ❌ every thread does full reduction
    }

    *result = sum;
}

int main() {
    int n = 1 << 16;
    size_t size = n * sizeof(float);

    float* h = new float[n];
    for (int i = 0; i < n; i++) {
        h[i] = 1.0f;
    }

    float *d, *d_result;
    cudaMalloc(&d, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    sumKernel<<<1, 1>>>(d, d_result, n);  // ❌ no parallelism at all

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << result << std::endl;

    delete[] h;
    cudaFree(d);
    cudaFree(d_result);
}
