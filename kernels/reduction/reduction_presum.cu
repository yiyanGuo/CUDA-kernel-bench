#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "reduction.h"

#define THREAD_PER_BLOCK 128
#define ELE_PER_THREAD 4
#define ELE_PER_BLOCK (THREAD_PER_BLOCK * ELE_PER_THREAD)

__global__ void kernel_reduction_presum(const float* input, float* output, int N) {
    __shared__ float buffer[THREAD_PER_BLOCK];
    const int tid = threadIdx.x;
    const int g_idx = blockIdx.x * ELE_PER_BLOCK + ELE_PER_THREAD * tid;

    float sum = .0f;
    for(int i = 0; i < ELE_PER_THREAD; i++) {
        if(g_idx + i < N) sum += input[g_idx + i];
    }
    buffer[tid] = sum;

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



void reduction_presum(const float* input, float* output, int N) {
    const float* d_in = input;
    float* d_out = nullptr;
    bool need_free = false;
    while(N > 1) {
        int threads = THREAD_PER_BLOCK;
        int blocks = (N + 4 * threads - 1) / (4 * threads);
        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
        kernel_reduction_presum<<<blocks, threads>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
        if(need_free) {
            CUDA_CHECK(cudaFree((void*)d_in));
        }
        need_free = true;
        d_in = d_out;
        N = blocks;
    }

    CUDA_CHECK(cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToDevice));
    if(need_free) CUDA_CHECK(cudaFree((void*)d_in));
}
