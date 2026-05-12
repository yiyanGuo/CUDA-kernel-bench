#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define SM_COUNT 114
#define BLOCK_PER_SM 4

#define THREAD_PER_BLOCK 128
#define ELE_PER_THREAD 4

__global__ void kernel_reduction_integrate(const float* input, float* output, int N) {
    __shared__ float buffer[THREAD_PER_BLOCK];
    constexpr int ele_per_block = THREAD_PER_BLOCK * ELE_PER_THREAD;
    const int tid = threadIdx.x;
    const int base_idx = blockIdx.x * ele_per_block + tid * ELE_PER_THREAD;
    const int grid_stride = gridDim.x * ele_per_block;
    
    float sum = 0.0f;
    for (int g_idx = base_idx; g_idx < N; g_idx += grid_stride) {
        if (g_idx + 3 < N) {
            float4 nums = *reinterpret_cast<const float4*>(&input[g_idx]);
            sum += nums.x + nums.y + nums.z + nums.w;
        } else {
            if (g_idx < N)     sum += input[g_idx];
            if (g_idx + 1 < N) sum += input[g_idx + 1];
            if (g_idx + 2 < N) sum += input[g_idx + 2];
            if (g_idx + 3 < N) sum += input[g_idx + 3];
        }
    }
    buffer[tid] = sum;
    __syncthreads();

    for (int stride = THREAD_PER_BLOCK / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        float val = buffer[tid] + buffer[tid + 32];
        const unsigned int mask = 0xffffffff;
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

void reduction_integrate(const float* input, float* output, int N) {
    float* buffer1 = nullptr;
    float* buffer2 = nullptr;

    const int ele_per_block = ELE_PER_THREAD * THREAD_PER_BLOCK;
    int blocks = min((N + ele_per_block - 1) / ele_per_block, SM_COUNT * BLOCK_PER_SM);

    CUDA_CHECK(cudaMalloc(&buffer1, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffer2, blocks * sizeof(float)));
    
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_reduction_integrate<<<blocks, THREAD_PER_BLOCK>>>(input, buffer1, N);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    N = blocks;
    float* d_in = buffer1;
    float* d_out = buffer2;
    float* tmp = nullptr;

    while (N > 1) {
        blocks = (N + ele_per_block - 1) / ele_per_block;
        kernel_reduction_integrate<<<blocks, THREAD_PER_BLOCK>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());

        tmp = d_in;
        d_in = d_out;
        d_out = tmp;
        N = blocks;
    }
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // printf("kernel time: %f\n", ms);

    CUDA_CHECK(cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(buffer1));
    CUDA_CHECK(cudaFree(buffer2));

}