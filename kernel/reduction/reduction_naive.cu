#include "cuda_utils.h"
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 128

__global__ void kernel_reduction_naive(const float* input, float* output, int N) {
    __shared__ float buffer[THREAD_PER_BLOCK];
    const int tid = threadIdx.x;
    const int gid = tid + blockIdx.x * THREAD_PER_BLOCK;
    
    buffer[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    for(int stride = THREAD_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if(tid < stride) {
            buffer[tid] = buffer[tid] + buffer[tid + stride];
        }
        // 每一层计算都应该同步
        __syncthreads();
    }

    const int bid = blockIdx.x;
    // 只需要一个线程写就可以了
    if(tid == 0)
        output[bid] = buffer[0];
}

void reduction_naive(const float* input, float* output, int N) {
    const float* d_in = input;
    float* d_out = nullptr;
    bool free = false;
    while(N > 1) {
        int threads = THREAD_PER_BLOCK;
        int blocks = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
        kernel_reduction_naive<<<blocks, threads>>>(d_in,d_out, N);
        CUDA_CHECK(cudaGetLastError());
        if(free) {
            CUDA_CHECK(cudaFree((void*)d_in));
        }
        free = true;
        d_in = d_out;
        N = blocks;
    }

    CUDA_CHECK(cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToDevice));
}