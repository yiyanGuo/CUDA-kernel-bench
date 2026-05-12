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

__global__ void kernel_scan_one_block(const float* input, float* output, int N) {
    __shared__ float thread_sums[THREAD_PER_BLOCK];
    __shared__ float carry;

    int tx = threadIdx.x;

    if (tx == 0) {
        carry = 0.0f;
    }

    __syncthreads();

    for (int seg = 0; seg < N; seg += BLOCK_SIZE) {
        int idx = seg + tx * ELE_PER_THREAD;

        float4 in = load_float4(input, idx, N);

        // exclusive scan inside one thread
        float local0 = 0.0f;
        float local1 = in.x;
        float local2 = in.x + in.y;
        float local3 = in.x + in.y + in.z;

        float thread_total = in.x + in.y + in.z + in.w;

        thread_sums[tx] = thread_total;
        __syncthreads();

        // inclusive scan over thread_sums
        for (int stride = 1; stride < THREAD_PER_BLOCK; stride <<= 1) {
            float add = 0.0f;

            if (tx >= stride) {
                add = thread_sums[tx - stride];
            }

            __syncthreads();

            if (tx >= stride) {
                thread_sums[tx] += add;
            }

            __syncthreads();
        }

        // exclusive offset for this thread
        float offset = carry;
        if (tx > 0) {
            offset += thread_sums[tx - 1];
        }

        if (idx + 0 < N) output[idx + 0] = offset + local0;
        if (idx + 1 < N) output[idx + 1] = offset + local1;
        if (idx + 2 < N) output[idx + 2] = offset + local2;
        if (idx + 3 < N) output[idx + 3] = offset + local3;

        __syncthreads();

        if (tx == 0) {
            carry += thread_sums[THREAD_PER_BLOCK - 1];
        }

        __syncthreads();
    }
}

void scan_one_block(const float* input, float* output, int N) {
    if (N <= 0) return;

    kernel_scan_one_block<<<1, THREAD_PER_BLOCK>>>(input, output, N);
}