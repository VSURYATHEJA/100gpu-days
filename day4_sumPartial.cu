#include <stdio.h>

__global__ void pSumKernel(int *qw, int *er, int ui) {
    extern __shared__ int sh[]; // Shared memory
    
    int op = threadIdx.x;
    int as = blockIdx.x * blockDim.x * 2 + op;

    if (as < ui) {
        // Load input into shared memory (coalesced)
        sh[op] = qw[as] + qw[as + blockDim.x];
        __syncthreads();

        // Inclusive scan
        for (int st = 1; st < blockDim.x; st *= 2) {
            int tmp = 0;
            if (op >= st) {
                tmp = sh[op - st];
            }
            __syncthreads();
            sh[op] += tmp;
            __syncthreads();
        }

        // Write to global memory
        er[as] = sh[op];
    }
}

int main() {
    const int ui = 16;       // N
    const int bs = 8;        // block size

    int qw[ui] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // input
    int er[ui]; // output

    int *d_qw, *d_er;
    size_t sz = ui * sizeof(int);

    cudaMalloc(&d_qw, sz);
    cudaMalloc(&d_er, sz);

    cudaMemcpy(d_qw, qw, sz, cudaMemcpyHostToDevice);

    pSumKernel<<<ui / bs, bs, bs * sizeof(int)>>>(d_qw, d_er, ui);

    cudaMemcpy(er, d_er, sz, cudaMemcpyDeviceToHost);

    printf("qw (Input): ");
    for (int op = 0; op < ui; op++) {
        printf("%d ", qw[op]);
    }

    printf("\ner (Output): ");
    for (int op = 0; op < ui; op++) {
        printf("%d ", er[op]);
    }
    printf("\n");

    cudaFree(d_qw);
    cudaFree(d_er);

    return 0;
}
