#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* qw, float* er, int ty, int ui) {
    int op = blockIdx.x; // block index → row number

    extern __shared__ float sh[]; // shared memory for the row

    if (op < ty) {
        int id = threadIdx.x;
        for (int p = id; p < ui; p += blockDim.x) {
            sh[p] = qw[op * ui + p];
        }
        __syncthreads();

        float rt = 0.0f; // mean
        for (int p = 0; p < ui; p++) {
            rt += sh[p];
        }
        rt /= ui;

        float yu = 0.0f; // variance
        for (int p = 0; p < ui; p++) {
            float df = sh[p] - rt;
            yu += df * df;
        }
        yu /= ui;
        float sd = sqrtf(yu + 1e-7);

        for (int p = id; p < ui; p += blockDim.x) {
            er[op * ui + p] = (sh[p] - rt) / sd;
        }
    }
}

int main() {
    const int ty = 10, ui = 10;
    float *qw, *er;

    // Allocate host memory
    qw = (float*)malloc(ty * ui * sizeof(float));
    er = (float*)malloc(ty * ui * sizeof(float));

    // Initialize matrix
    for (int i = 0; i < ty; i++) {
        for (int j = 0; j < ui; j++) {
            qw[i * ui + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Device memory
    float *d_qw, *d_er;
    cudaMalloc(&d_qw, ty * ui * sizeof(float));
    cudaMalloc(&d_er, ty * ui * sizeof(float));

    cudaMemcpy(d_qw, qw, ty * ui * sizeof(float), cudaMemcpyHostToDevice);

    int blk = 256;
    size_t shmem = ui * sizeof(float);
    LayerNorm<<<ty, blk, shmem>>>(d_qw, d_er, ty, ui);
    cudaDeviceSynchronize();

    cudaMemcpy(er, d_er, ty * ui * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("qw (Input Matrix):\n");
    for (int i = 0; i < ty; i++) {
        for (int j = 0; j < ui; j++) {
            printf("%.2f ", qw[i * ui + j]);
        }
        printf("\n");
    }

    printf("\ner (Normalized Output):\n");
    for (int i = 0; i < ty; i++) {
        for (int j = 0; j < ui; j++) {
            printf("%.2f ", er[i * ui + j]);
        }
        printf("\n");
    }

    cudaFree(d_qw);
    cudaFree(d_er);
    free(qw);
    free(er);

    return 0;
}
