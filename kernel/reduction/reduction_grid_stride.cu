#include "cuda_runtime.h"
#include "cuda_utils.h"


#define SM_COUNT 114
#define BLOCK_PER_SM 8

#define THREAD_PER_BLOCK 128
#define ELE_PER_THREAD 1
#define ELE_PER_BLOCK (THREAD_PER_BLOCK * ELE_PER_THREAD)


__global__ void kernel_reduction_grid_stride(const float* input, float* output, int N) {
    __shared__ float buffer[THREAD_PER_BLOCK];
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * ELE_PER_BLOCK;
    const int base_idx = blockIdx.x * ELE_PER_BLOCK + tid;
    
    // presum
    float sum = 0.0f;
#pragma unroll
    for(int i = base_idx; i < N; i += grid_stride) {
        sum += input[i];
    }
    buffer[tid] = sum;

    __syncthreads();

    // reduce
    for(int stride = THREAD_PER_BLOCK / 2; stride > 32; stride /= 2) {
        if(tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }

    // tail shuffle
    if(tid < 32) {
        constexpr unsigned int mask = 0xffffffff;
        float val = buffer[tid] + buffer[tid + 32];
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if(tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

void reduction_grid_stride(const float* input, float* output, int N) {
    const float* d_in = input;
    float* d_out = nullptr;
    constexpr int blocks = SM_COUNT * BLOCK_PER_SM;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * blocks));
    kernel_reduction_grid_stride<<<blocks, THREAD_PER_BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    N = blocks;
    d_in = d_out;
    d_out = nullptr;

    while(N > 1) {
        int gridDim = (N + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * gridDim));
        kernel_reduction_grid_stride<<<gridDim, THREAD_PER_BLOCK>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree((void*)d_in));
        d_in = d_out;
        d_out = nullptr;
        N = gridDim;
    }

    CUDA_CHECK(cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree((void*)d_in));
}