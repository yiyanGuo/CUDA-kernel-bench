#include "cuda_runtime.h"
#include "cuda_utils.h"

#include "rms_norm.h"

#define THREAD_PER_BLOCK 256
#define WARPS (THREAD_PER_BLOCK / 32)



__global__ void kernel_rms_naive(const float *input, const float *weight,
                                 float *output, int num_tokens, int hidden_size,
                                 float eps) {
    __shared__ float inv_rms;
    __shared__ float sum_sq;
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if(row >= num_tokens) {
        return;
    }
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const unsigned int mask = 0xffffffff;
    
    if(tid == 0) {
        sum_sq = 0.0f;
    }
    __syncthreads();

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

    // sum in warp
#pragma unroll
    for(int offset = 16; offset >= 1; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if(lane_id == 0){
        atomicAdd(&sum_sq, local_sum);
    }
    __syncthreads();

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

void rms_naive(const float *input, const float *weight, float *output,
               int num_tokens, int hidden_size, float eps) {
    int blocks = num_tokens;
    kernel_rms_naive<<<blocks, THREAD_PER_BLOCK>>>(input, weight, output, num_tokens, hidden_size, eps);
    CUDA_CHECK(cudaGetLastError());
}
