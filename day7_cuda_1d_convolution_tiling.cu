#include <iostream>
#include <cuda_runtime.h>

#define KERNEL_SIZE 5
__constant__ float d_kernel[KERNEL_SIZE]; // Convolution kernel in constant memory

// CUDA Kernel for 1D convolution using tiling
__global__ void convolution1DTiling(const float* d_input, float* d_output, int length) {
    int threadId = threadIdx.x;
    int globalIdx = blockDim.x * blockIdx.x + threadId;

    __shared__ float sharedTile[32 + KERNEL_SIZE - 1]; // Tile with halo

    // Load main data
    if (globalIdx < length) {
        sharedTile[threadId + KERNEL_SIZE / 2] = d_input[globalIdx];
    }

    // Load left halo
    if (threadId < KERNEL_SIZE / 2) {
        int leftIdx = blockIdx.x * blockDim.x - (KERNEL_SIZE / 2) + threadId;
        sharedTile[threadId] = (leftIdx >= 0) ? d_input[leftIdx] : 0.0f;
    }

    // Load right halo
    if (threadId < KERNEL_SIZE / 2) {
        int rightIdx = blockIdx.x * blockDim.x + blockDim.x + threadId;
        sharedTile[threadId + blockDim.x + KERNEL_SIZE / 2] =
            (rightIdx < length) ? d_input[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Compute convolution
    if (globalIdx < length) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            int tileIdx = threadId + k;
            if ((globalIdx + k - KERNEL_SIZE / 2) >= 0 &&
                (globalIdx + k - KERNEL_SIZE / 2) < length) {
                sum += sharedTile[tileIdx] * d_kernel[k];
            }
        }
        d_output[globalIdx] = sum;
    }
}

// Function to check CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int length = 10;
    float h_input[length], h_output[length], h_kernel[KERNEL_SIZE];

    // Initialize kernel values
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        h_kernel[i] = static_cast<float>(i);
    }

    // Initialize input values
    for (int i = 0; i < length; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, length * sizeof(float));
    cudaMalloc(&d_output, length * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, length * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));
    checkCudaError("Failed to copy kernel to device");

    // Kernel configuration
    dim3 blockDim(32);
    dim3 gridDim((length + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    convolution1DTiling<<<gridDim, blockDim>>>(d_input, d_output, length);
    checkCudaError("Kernel execution failed");

    cudaDeviceSynchronize();

    // Copy output to host
    cudaMemcpy(h_output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Print results
    std::cout << "Input Array:\n";
    for (int i = 0; i < length; ++i) std::cout << h_input[i] << " ";
    std::cout << "\n\nKernel:\n";
    for (int i = 0; i < KERNEL_SIZE; ++i) std::cout << h_kernel[i] << " ";
    std::cout << "\n\nOutput Array:\n";
    for (int i = 0; i < length; ++i) std::cout << h_output[i] << " ";
    std::cout << "\n";

    return 0;
}
