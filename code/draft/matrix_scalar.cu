#include <cuda_runtime.h>
#include <iostream>

__global__ void scaleMatrix(float* m, float scalar, int n) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    int idx = row * n + col;
    if (idx < n * n) {
        m[idx] *= scalar;
    }
}

int main() {
    int n = 512;
    size_t size = n * n * sizeof(float);

    float* h = new float[n * n];
    for (int i = 0; i < n * n; i++) {
        h[i] = 1.0f;
    }

    float* d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    scaleMatrix<<<n, n>>>(d, 3.0f, n);  // ❌ massive blocks

    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    std::cout << h[0] << std::endl;

    delete[] h;
    cudaFree(d);
}
