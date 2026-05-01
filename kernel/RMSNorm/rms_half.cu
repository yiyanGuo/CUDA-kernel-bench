#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "cuda_fp16.h"
#include <cuda_runtime_api.h>

#define THREAD_PER_BLOCK 256
#define WARPS (THREAD_PER_BLOCK / 32)


__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sum[WARPS];

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_sum(val);

    if(lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    val = (tid < WARPS) ? warp_sum[lane_id] : 0.0f;

    if(warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__global__ void kernel_rms_half(const half* input, const half* weight, half* output, 
                                int num_tokens, int hidden_size, 
                                float eps) {
    __shared__ float inv_rms;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if(bid >= num_tokens) {
        return;
    }
    
    // thread pre sum
    float local_sum = 0.0f;
    for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
        int col = tid + seg;
        if(col < hidden_size) {
            float val = __half2float(input[bid * hidden_size + col]);
            local_sum += val * val;
        }
    }

    // only thread 0 has right val
    float sum_sq = block_reduce_sum(local_sum);

    if(tid == 0) {
        inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // second pass normalize
    for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        if(col < hidden_size) {
            float x = __half2float(input[bid * hidden_size + col]);
            float w = __half2float(weight[col]);
            float y = x * inv_rms * w;
            output[bid * hidden_size + col] = __float2half(y);
        }
    }
}

void rms_half(const half* input, const half* weight, half* output, 
            int num_tokens, int hidden_size,
            float eps) {
    int blocks = num_tokens;
    kernel_rms_half<<<blocks, THREAD_PER_BLOCK>>>(input, weight, output, num_tokens, hidden_size, eps);
    CUDA_CHECK(cudaGetLastError());
}
