#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void helloWorldKernel() {
    printf("Hello, World! from thread %d\n", threadIdx.x);
}

// Function to launch the kernel
void launchKernel(int threadsPerBlock) {
    helloWorldKernel<<<1, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // Wait for kernel to finish
}

// The main function from the host
int main() {
    int threadsPerBlock = 4;
    launchKernel(threadsPerBlock);
    printf("Hello, World! from the host\n");
    return 0;
}
