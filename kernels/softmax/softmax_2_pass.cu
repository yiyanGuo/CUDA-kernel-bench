#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <cmath>
#include <cuda_runtime_api.h>
#include <vector_types.h>

#define THREAD_PER_BLOCK 128
#define WARPS (THREAD_PER_BLOCK / 32)

__device__ __forceinline__ float2 warp_reduce(float thread_max, float thread_sum) {
    // max
    float lane_max = thread_max;
#pragma unroll
    for(int offset = 16; offset >= 1; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    float max = __shfl_sync(0xffffffff, thread_max, 0);

    if (max == -INFINITY) {
        return make_float2(-INFINITY, 0.0f);
    }
    // sum
    thread_sum = thread_sum * expf(lane_max - max); // sun re-scale
#pragma unroll
    for(int offset = 16; offset >= 1; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    float sum = __shfl_sync(0xffffffff, thread_sum, 0);

    float2 res = make_float2(max, sum);
    return res;
}

__device__ __forceinline__ float2 block_reduce(float thread_max, float thread_sum) {
    __shared__ float warp_sums[WARPS];
    __shared__ float warp_maxs[WARPS];
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    float2 warp_res = warp_reduce(thread_max, thread_sum);
    if(lane_id == 0) {
        warp_maxs[warp_id] = warp_res.x;
        warp_sums[warp_id] = warp_res.y;
    }
    __syncthreads();

    // block reduce
    if(warp_id == 0) {
        float warp_sum = lane_id < WARPS ? warp_sums[lane_id] : 0.0f;
        float warp_max = lane_id < WARPS ? warp_maxs[lane_id] : -INFINITY;
        float2 res = warp_reduce(warp_max, warp_sum);
        if(lane_id == 0) {
            warp_maxs[0] = res.x;
            warp_sums[0] = res.y;
        }
    }
    __syncthreads();
    
    float2 res = make_float2(warp_maxs[0], warp_sums[0]);
    return res;
} 

__global__ void kernel_softmax_2_pass(
    const float* logits,
    float* output, 
    int batch_size,
    int num_heads,
    int query_len,
    int key_len
) {
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int qid = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = ((batch * num_heads + head) * query_len + qid) * key_len;
    const float* logits_row = logits + row_offset;
    float* output_row = output + row_offset;

    // max-sum
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for(int idx = tid; idx < key_len; idx += THREAD_PER_BLOCK) {
        float x = logits_row[idx];
        float new_local_max = fmaxf(local_max, x);
        local_sum = local_sum * expf(local_max - new_local_max) + expf(x - new_local_max);
        local_max = new_local_max;
    }
    __syncthreads();

    float2 max_sum = block_reduce(local_max, local_sum);

    // softmax
    for(int idx = tid; idx < key_len; idx += THREAD_PER_BLOCK) {
        float x = logits_row[idx];
        output_row[idx] = expf(x - max_sum.x) / max_sum.y;
    }
}

void softmax_2_pass(
    const float* logits,
    float* output, 
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    bool casual
) {
    dim3 gridDim(query_len, num_heads, batch_size);
    dim3 blockDim(THREAD_PER_BLOCK);
    kernel_softmax_2_pass<<<gridDim, blockDim>>>(
        logits,
        output,
        batch_size,
        num_heads,
        query_len,
        key_len
    );
    CUDA_CHECK(cudaGetLastError());
}