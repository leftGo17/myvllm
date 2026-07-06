#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" {

// includes host<->device copy time, so the comparison is apples-to-apples
// with the cpp/python versions which also start and end with host memory
void add_arrays_cuda(const float* a, const float* b, float* out, int n) {
    float *d_a, *d_b, *d_out;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

}
