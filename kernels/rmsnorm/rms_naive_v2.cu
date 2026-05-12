#include "cuda_runtime.h"
#include "cuda_utils.h"

#include "rms_norm.h"

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

__global__ void kernel_rms_naive_v2(const float *input, const float *weight,
                                    float *output, int num_tokens,
                                    int hidden_size, float eps) {
    __shared__ float inv_rms;
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if(row >= num_tokens) {
        return;
    }
    
    // first-pass get rms
    float local_sum = 0.0f;
    for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        float val = 0.0f;
        if(col < hidden_size) {
            val = input[row * hidden_size + col];
        }
        local_sum += val * val;
    }

    float sum_sq = block_reduce_sum(local_sum);

    if(tid == 0) {
        inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    // second pass normalize
    for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        if(col < hidden_size) {
            output[row * hidden_size + col] =
                input[row * hidden_size + col] * inv_rms * weight[col];
        }
    }
}

void rms_naive_v2(const float *input, const float *weight, float *output,
                  int num_tokens, int hidden_size, float eps) {
    int blocks = num_tokens;
    kernel_rms_naive_v2<<<blocks, THREAD_PER_BLOCK>>>(input, weight, output,
                                                      num_tokens, hidden_size,
                                                      eps);
    CUDA_CHECK(cudaGetLastError());
}
