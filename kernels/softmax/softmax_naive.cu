#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <cmath>
#include <cuda_runtime_api.h>

#define THREAD_PER_BLOCK 128
#define WARPS (THREAD_PER_BLOCK / 32)

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) {
        return a > b ? a : b;
    }
};

struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) {
        return a + b;
    }
};

template <typename Op>
__device__ __forceinline__ float warp_reduce(float val, Op op) {
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename Op>
__device__ float block_reduce(float thread_val, float identity, Op op) {
    __shared__ float warp_reduction[WARPS];
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    thread_val = warp_reduce(thread_val, op);

    if(lane_id == 0) {
        warp_reduction[warp_id] = thread_val;
    }
    __syncthreads();

    if(warp_id == 0) {
        thread_val = (tid < WARPS) ? warp_reduction[lane_id] : identity;
        thread_val = warp_reduce(thread_val, op);
    }

    if(tid == 0) {
        warp_reduction[0] = thread_val;
    }
    __syncthreads();

    // 每个线程都能拿到正确结果
    return warp_reduction[0];
}

__global__ void kernel_softmax_naive(
    const float* logits,
    float* output, 
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    bool casual
) {
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int qid = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = ((batch * num_heads + head) * query_len + qid) * key_len;
    const float* logits_row = logits + row_offset;
    float* output_row = output + row_offset;

    // mask
    const int valid_key_len = casual ? min(qid + 1, key_len) : key_len;
    // max
    float local_max = -INFINITY;
    for(int idx = tid; idx < valid_key_len; idx += THREAD_PER_BLOCK) {
        local_max = fmaxf(local_max, logits_row[idx]);
    }

    local_max = block_reduce(local_max, -INFINITY,MaxOp{});
    
    // exp sum
    float local_sum = 0.0f;
    for(int idx = tid; idx < valid_key_len; idx += THREAD_PER_BLOCK) {
        local_sum = local_sum + expf(logits_row[idx] - local_max);
    }
    local_sum = block_reduce(local_sum, 0.0f,SumOp{});

    // softmax
    for(int idx = tid; idx < valid_key_len; idx += THREAD_PER_BLOCK) {
        output_row[idx] = expf(logits_row[idx] - local_max) / local_sum;
    }
    for(int idx = valid_key_len + tid; idx < key_len; idx += THREAD_PER_BLOCK) {
        output_row[idx] = 0.0f;
    }
}

void softmax_naive(
    const float* logits,
    float* output, 
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    bool casual
) {
    dim3 gridDim(query_len, num_heads, batch_size);
    dim3 blockDim(128);
    kernel_softmax_naive<<<gridDim, blockDim>>>(
        logits,
        output,
        batch_size,
        num_heads,
        query_len,
        key_len,
        casual
    );
    CUDA_CHECK(cudaGetLastError());
}
