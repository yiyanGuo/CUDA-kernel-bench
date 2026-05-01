#include "cuda_runtime.h"
#include "cuda_utils.h"

#include "rms_norm.h"

#define THREAD_PER_BLOCK 256
#define WARPS (THREAD_PER_BLOCK / 32)

template <int HIDDEN_SIZE>
__global__ void kernel_rms_shared_memory(const float *input, const float *weight,
                                         float *output, int num_tokens,
                                         float eps) {
    __shared__ float row_data[HIDDEN_SIZE];
    __shared__ float inv_rms;
    __shared__ float sum_sq;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if(row >= num_tokens) {
        return;
    }
    const int lane_id = tid % 32;
    const unsigned int mask = 0xffffffff;

    if(tid == 0) {
        sum_sq = 0.0f;
    }
    __syncthreads();

    for(int seg = 0; seg < HIDDEN_SIZE; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        if(col < HIDDEN_SIZE) {
            row_data[col] = input[row * HIDDEN_SIZE + col];
        }
    }
    __syncthreads();

    float local_sum = 0.0f;
    for(int seg = 0; seg < HIDDEN_SIZE; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        if(col < HIDDEN_SIZE) {
            float val = row_data[col];
            local_sum += val * val;
        }
    }

#pragma unroll
    for(int offset = 16; offset >= 1; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if(lane_id == 0) {
        atomicAdd(&sum_sq, local_sum);
    }
    __syncthreads();

    if(tid == 0) {
        inv_rms = rsqrtf(sum_sq / HIDDEN_SIZE + eps);
    }
    __syncthreads();

    for(int seg = 0; seg < HIDDEN_SIZE; seg += THREAD_PER_BLOCK) {
        int col = seg + tid;
        if(col < HIDDEN_SIZE) {
            output[row * HIDDEN_SIZE + col] =
                row_data[col] * inv_rms * weight[col];
        }
    }
}

template <int HIDDEN_SIZE>
void launch_rms_shared_memory(const float *input, const float *weight,
                              float *output, int num_tokens, float eps) {
    kernel_rms_shared_memory<HIDDEN_SIZE><<<num_tokens, THREAD_PER_BLOCK>>>(
        input, weight, output, num_tokens, eps);
}

void rms_shared_memory(const float *input, const float *weight, float *output,
                       int num_tokens, int hidden_size, float eps) {
    switch(hidden_size) {
    case 256:
        launch_rms_shared_memory<256>(input, weight, output, num_tokens, eps);
        break;
    case 512:
        launch_rms_shared_memory<512>(input, weight, output, num_tokens, eps);
        break;
    case 1024:
        launch_rms_shared_memory<1024>(input, weight, output, num_tokens, eps);
        break;
    case 2048:
        launch_rms_shared_memory<2048>(input, weight, output, num_tokens, eps);
        break;
    case 4096:
        launch_rms_shared_memory<4096>(input, weight, output, num_tokens, eps);
        break;
    case 8192:
        launch_rms_shared_memory<8192>(input, weight, output, num_tokens, eps);
        break;
    default:
        std::fprintf(stderr,
                     "Unsupported hidden_size %d for rms_shared_memory.\n",
                     hidden_size);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
}
