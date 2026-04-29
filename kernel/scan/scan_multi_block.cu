#include "cuda_runtime.h"
#include "cuda_utils.h"

#define THREAD_PER_BLOCK 256
#define ELE_PER_THREAD 4
#define BLOCK_SIZE (ELE_PER_THREAD * THREAD_PER_BLOCK)

__device__ __forceinline__
float4 load_float4(const float* input, int idx, int N) {
    float4 v;

    if (idx + 3 < N) {
        v = *reinterpret_cast<const float4*>(input + idx);
    } else {
        v.x = (idx + 0 < N) ? input[idx + 0] : 0.0f;
        v.y = (idx + 1 < N) ? input[idx + 1] : 0.0f;
        v.z = (idx + 2 < N) ? input[idx + 2] : 0.0f;
        v.w = (idx + 3 < N) ? input[idx + 3] : 0.0f;
    }

    return v;
}

__global__ void kernel_scan_in_block(const float* input, float* output, int N, float* block_sum) {
    // thread_sums是 inclusive 的
    __shared__ float thread_sums[THREAD_PER_BLOCK];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = gid * ELE_PER_THREAD;

    float4 in = load_float4(input, idx, N);
    float local0 = 0.0f;
    float local1 = in.x;
    float local2 = in.x + in.y;
    float local3 = in.x + in.y + in.z;

    float thread_total = in.x + in.y + in.z + in.w;
    thread_sums[tid] = thread_total;
    __syncthreads();

    // scan on thread_sums
    for(int stride = 1; stride < THREAD_PER_BLOCK; stride *= 2) {
        float add = 0.0f;
        if(tid >= stride) {
            add = thread_sums[tid - stride];
        }
        __syncthreads();
        if(tid >= stride) {
            thread_sums[tid] += add;
        }
        __syncthreads();
    }

    // add offset
    float offset = 0.0f;
    if(tid > 0) {
        offset = thread_sums[tid - 1];
    }

    if (idx + 0 < N) output[idx + 0] = offset + local0;
    if (idx + 1 < N) output[idx + 1] = offset + local1;
    if (idx + 2 < N) output[idx + 2] = offset + local2;
    if (idx + 3 < N) output[idx + 3] = offset + local3;  

    if(tid == blockDim.x - 1) {
        block_sum[blockIdx.x] = thread_sums[THREAD_PER_BLOCK - 1];
    }
}

__global__ void kernel_add_offset_multi_block(float* data, float* offset, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx = tid * ELE_PER_THREAD;
    const int bid = blockIdx.x;
    float4 old = load_float4(data, idx, N);
    float add = 0.0f;
    add = offset[bid];

    if (idx + 0 < N) data[idx + 0] = add + old.x;
    if (idx + 1 < N) data[idx + 1] = add + old.y;
    if (idx + 2 < N) data[idx + 2] = add + old.z;
    if (idx + 3 < N) data[idx + 3] = add + old.w; 
}

void scan_multi_block(const float* input, float* output, int N) {
    if (N == 1) {
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        return;
    }
    
    int threads = THREAD_PER_BLOCK;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* block_sums;
    CUDA_CHECK(cudaMalloc(&block_sums, sizeof(float) * blocks));
    kernel_scan_in_block<<<blocks, threads>>>(input, output, N, block_sums);
    CUDA_CHECK(cudaGetLastError());

    scan_multi_block(block_sums, block_sums, blocks);

    kernel_add_offset_multi_block<<<blocks, threads>>>(output, block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree((void*)block_sums));
}