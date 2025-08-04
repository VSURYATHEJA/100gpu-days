#include <stdio.h>
#include <iostream>

#define KERNEL_SIZE 5
#define SHARED_MEM_SIZE (32 + KERNEL_SIZE - 1)
__constant__ float filterKernel[KERNEL_SIZE][KERNEL_SIZE];

__global__ void convolution2DSharedKernel(const float* inputMatrix, float* outputMatrix, int matrixSize) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.x * blockIdx.x + tx;
    int col = blockDim.y * blockIdx.y + ty;

    __shared__ float sharedTile[SHARED_MEM_SIZE][SHARED_MEM_SIZE];

    // Load main data into shared memory
    if ((row < matrixSize) && (col < matrixSize)) {
        sharedTile[tx + KERNEL_SIZE / 2][ty + KERNEL_SIZE / 2] = inputMatrix[row * matrixSize + col];
    }

    // Left halo
    if (tx < KERNEL_SIZE / 2) {
        int leftIndex = blockIdx.x * blockDim.x - (KERNEL_SIZE / 2) + tx;
        sharedTile[tx][ty + KERNEL_SIZE / 2] = (leftIndex >= 0 && col < matrixSize) 
                                               ? inputMatrix[leftIndex * matrixSize + col] 
                                               : 0.0f;
    }

    // Right halo
    if (tx < KERNEL_SIZE / 2) {
        int rightIndex = blockIdx.x * blockDim.x + blockDim.x + tx;
        sharedTile[tx + blockDim.x + KERNEL_SIZE / 2][ty + KERNEL_SIZE / 2] = 
            (rightIndex < matrixSize && col < matrixSize) 
            ? inputMatrix[rightIndex * matrixSize + col] 
            : 0.0f;
    }

    // Top halo
    if (ty < KERNEL_SIZE / 2) {
        int topIndex = col - (KERNEL_SIZE / 2) + ty;
        sharedTile[tx + KERNEL_SIZE / 2][ty] = (topIndex >= 0 && row < matrixSize) 
                                               ? inputMatrix[row * matrixSize + topIndex] 
                                               : 0.0f;
    }

    // Bottom halo
    if (ty < KERNEL_SIZE / 2) {
        int bottomIndex = col + blockDim.y + ty;
        sharedTile[tx + KERNEL_SIZE / 2][ty + blockDim.y + KERNEL_SIZE / 2] = 
            (bottomIndex < matrixSize && row < matrixSize) 
            ? inputMatrix[row * matrixSize + bottomIndex] 
            : 0.0f;
    }

    __syncthreads();

    if ((row < matrixSize) && (col < matrixSize)) {
        float sum = 0.0f;
        for (int m = 0; m < KERNEL_SIZE; m++) {
            for (int n = 0; n < KERNEL_SIZE; n++) {
                sum += sharedTile[tx + m][ty + n] * filterKernel[m][n];
            }
        }
        outputMatrix[row * matrixSize + col] = sum;
    }
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "%s - CUDA Error: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int matrixSize = 10;
    float *h_input = (float*)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_output = (float*)malloc(matrixSize * matrixSize * sizeof(float));
    float h_filter[KERNEL_SIZE][KERNEL_SIZE];

    // Fill kernel with constant value (example: 5)
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            h_filter[i][j] = 5.0f;
        }
    }

    // Fill input matrix with constant value (example: 3)
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            h_input[i * matrixSize + j] = 3.0f;
        }
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, matrixSize * matrixSize * sizeof(float));
    cudaMalloc(&d_output, matrixSize * matrixSize * sizeof(float));

    cudaMemcpy(d_input, h_input, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input matrix to device");

    cudaMemcpyToSymbol(filterKernel, h_filter, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    checkCudaError("Failed to copy filter to constant memory");

    dim3 blockDim(32, 32);
    dim3 gridDim((matrixSize + blockDim.x - 1) / blockDim.x, (matrixSize + blockDim.y - 1) / blockDim.y);

    convolution2DSharedKernel<<<gridDim, blockDim>>>(d_input, d_output, matrixSize);
    checkCudaError("Kernel execution failed");

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output matrix to host");

    // Print output matrix
    printf("Output Matrix:\n");
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%.2f ", h_output[i * matrixSize + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
