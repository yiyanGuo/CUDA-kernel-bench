#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "reduction_cu.cuh"

#define THREAD_PER_BLOCK 128
#define ELE_PER_THREAD 4
#define ELE_PER_BLOCK (THREAD_PER_BLOCK * ELE_PER_THREAD)

__global__ void kernel_reduction_presum_float4(const float* input, float* output, int N) {
    __shared__ float buffer[THREAD_PER_BLOCK];

    const int tid = threadIdx.x;
    const int g_idx = blockIdx.x * ELE_PER_BLOCK + ELE_PER_THREAD * tid;

    float sum = 0.0f;

    if (g_idx + 3 < N) {
        float4 nums = *reinterpret_cast<const float4*>(&input[g_idx]);
        sum = nums.x + nums.y + nums.z + nums.w;
    } else {
        if (g_idx < N)     sum += input[g_idx];
        if (g_idx + 1 < N) sum += input[g_idx + 1];
        if (g_idx + 2 < N) sum += input[g_idx + 2];
        if (g_idx + 3 < N) sum += input[g_idx + 3];
    }

    buffer[tid] = sum;
    __syncthreads();

    for (int stride = THREAD_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = buffer[0];
    }
}

void reduction_presum_float4(const float* input, float* output, int N) {
    const float* d_in = input;
    float* d_out = nullptr;
    bool need_free = false;

    while (N > 1) {
        int threads = THREAD_PER_BLOCK;
        int blocks = (N + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;

        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
        if(need_free == false) kernel_reduction_presum_float4<<<blocks, threads>>>(d_in, d_out, N);
        else kernel_reduction_presum<<<blocks, threads>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());

        if (need_free) {
            CUDA_CHECK(cudaFree((void*)d_in));
        }

        need_free = true;
        d_in = d_out;
        d_out = nullptr;
        N = blocks;
    }

    CUDA_CHECK(cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToDevice));

    if (need_free) {
        CUDA_CHECK(cudaFree((void*)d_in));
    }
}