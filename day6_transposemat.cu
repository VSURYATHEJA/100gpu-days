#include <cuda_runtime.h>
#include <iostream>

#define aa 1024  // width
#define bb 1024  // height

// CUDA kernel for matrix transpose
__global__ void tt(const float* xx, float* yy, int m, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (p < m && q < n) {
        int ii = q * m + p;
        int jj = p * n + q;
        yy[jj] = xx[ii];
    }
}

// Error check
void ee(const char* mm) {
    cudaError_t er = cudaGetLastError();
    if (er != cudaSuccess) {
        std::cerr << mm << " - CUDA Error: " << cudaGetErrorString(er) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int m = aa;
    int n = bb;

    size_t sz = m * n * sizeof(float);
    float* xx = (float*)malloc(sz); // input matrix
    float* yy = (float*)malloc(sz); // output matrix

    for (int i = 0; i < m * n; i++) {
        xx[i] = static_cast<float>(i);
    }

    float* dx;
    float* dy;
    cudaMalloc((void**)&dx, sz);
    cudaMalloc((void**)&dy, sz);

    cudaMemcpy(dx, xx, sz, cudaMemcpyHostToDevice);
    ee("Copy input");

    dim3 bbk(32, 32);
    dim3 ggr((m + bbk.x - 1) / bbk.x, (n + bbk.y - 1) / bbk.y);

    tt<<<ggr, bbk>>>(dx, dy, m, n);
    cudaDeviceSynchronize();
    ee("Kernel");

    cudaMemcpy(yy, dy, sz, cudaMemcpyDeviceToHost);
    ee("Copy output");

    bool ok = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (yy[i * n + j] != xx[j * m + i]) {
                ok = false;
                break;
            }
        }
    }

    std::cout << (ok ? "Transpose OK!" : "Transpose FAIL!") << std::endl;

    cudaFree(dx);
    cudaFree(dy);
    free(xx);
    free(yy);

    return 0;
}
