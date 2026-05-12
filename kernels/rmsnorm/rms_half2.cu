#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "cuda_fp16.h"
#include <cuda_runtime_api.h>

#define THREAD_PER_BLOCK 256
#define WARPS (THREAD_PER_BLOCK / 32)


__device__ __forceinline__ float warp_reduce_sum_half2(float val) {
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_half2(float val) {
    __shared__ float warp_sum[WARPS];

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_sum_half2(val);

    if(lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    val = (tid < WARPS) ? warp_sum[lane_id] : 0.0f;

    if(warp_id == 0) {
        val = warp_reduce_sum_half2(val);
    }

    return val;
}

__global__ void kernel_rms_half2(const half* input, const half* weight,
                                 half* output, int num_tokens,
                                 int hidden_size, float eps) {
    __shared__ float inv_rms;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if(bid >= num_tokens) {
        return;
    }

    const half* row_input = input + bid * hidden_size;
    half* row_output = output + bid * hidden_size;
    const bool use_half2 = (hidden_size % 2) == 0;

    // thread pre sum
    float local_sum = 0.0f;
    if(use_half2) {
        const int hidden_size_half2 = hidden_size / 2;
        const half2* row_input_half2 =
            reinterpret_cast<const half2*>(row_input);

        for(int seg = 0; seg < hidden_size_half2; seg += THREAD_PER_BLOCK) {
            int col = tid + seg;
            if(col < hidden_size_half2) {
                half2 val = row_input_half2[col];
                float x = __low2float(val);
                float y = __high2float(val);
                local_sum += x * x + y * y;
            }
        }
    } else {
        for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
            int col = tid + seg;
            if(col < hidden_size) {
                float val = __half2float(row_input[col]);
                local_sum += val * val;
            }
        }
    }

    // only thread 0 has right val
    float sum_sq = block_reduce_sum_half2(local_sum);

    if(tid == 0) {
        inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // second pass normalize
    if(use_half2) {
        const int hidden_size_half2 = hidden_size / 2;
        const half2* row_input_half2 =
            reinterpret_cast<const half2*>(row_input);
        const half2* weight_half2 = reinterpret_cast<const half2*>(weight);
        half2* row_output_half2 = reinterpret_cast<half2*>(row_output);

        for(int seg = 0; seg < hidden_size_half2; seg += THREAD_PER_BLOCK) {
            int col = seg + tid;
            if(col < hidden_size_half2) {
                half2 x2 = row_input_half2[col];
                half2 w2 = weight_half2[col];
                float x0 = __low2float(x2);
                float x1 = __high2float(x2);
                float w0 = __low2float(w2);
                float w1 = __high2float(w2);
                row_output_half2[col] = __floats2half2_rn(
                    x0 * inv_rms * w0, x1 * inv_rms * w1);
            }
        }
    } else {
        for(int seg = 0; seg < hidden_size; seg += THREAD_PER_BLOCK) {
            int col = seg + tid;
            if(col < hidden_size) {
                float x = __half2float(row_input[col]);
                float w = __half2float(weight[col]);
                float y = x * inv_rms * w;
                row_output[col] = __float2half(y);
            }
        }
    }
}

void rms_half2(const half* input, const half* weight, half* output,
               int num_tokens, int hidden_size, float eps) {
    int blocks = num_tokens;
    kernel_rms_half2<<<blocks, THREAD_PER_BLOCK>>>(input, weight, output,
                                                   num_tokens, hidden_size,
                                                   eps);
    CUDA_CHECK(cudaGetLastError());
}
