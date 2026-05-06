#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <cmath>
#include <cuda_runtime_api.h>

#define THREAD_PER_BLOCK 128
#define WARPS (THREAD_PER_BLOCK / 32)
#define BLOCK_M 32
#define BLOCK_N 32
#define HEAD_DIM 64


__global__ void kernel_flash_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len
) {
    __shared__ float q_smem[BLOCK_M * HEAD_DIM];
    __shared__ float o_smem[BLOCK_M * HEAD_DIM];
    __shared__ float k_smem[BLOCK_N * HEAD_DIM];
    __shared__ float v_smem[BLOCK_N * HEAD_DIM];
    __shared__ float l_smem[BLOCK_M];
    __shared__ float m_smem[BLOCK_M];

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int q_block_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const float* g_k_base = k + batch * (num_heads * key_len * HEAD_DIM) + head * (key_len * HEAD_DIM);
    const float* g_v_base = v + batch * (num_heads * key_len * HEAD_DIM) + head * (key_len * HEAD_DIM);
    const float* g_q_block = q + batch * (num_heads * query_len * HEAD_DIM) + head * (query_len * HEAD_DIM) + q_block_id * BLOCK_M * HEAD_DIM;
    float* g_o_block = output + batch * (num_heads * query_len * HEAD_DIM) + head * (query_len * HEAD_DIM) + q_block_id * BLOCK_M * HEAD_DIM;

    // init
    #pragma unroll
    for(int idx = tid; idx < BLOCK_M; idx += THREAD_PER_BLOCK) {
        l_smem[idx] = 0.0f;
        m_smem[idx] = -INFINITY;
    }
    #pragma unroll
    for(int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += THREAD_PER_BLOCK) {
        o_smem[idx] = 0.0f;
    }

    // lood Q
    #pragma unroll
    for(int idx = tid; idx < HEAD_DIM * BLOCK_M; idx += THREAD_PER_BLOCK) {
        q_smem[idx] = g_q_block[idx];
    }

    // K V loop
    for(int offset = 0; offset < key_len * HEAD_DIM; offset += HEAD_DIM * BLOCK_N) {
        const float* g_k_block = g_k_base + offset;
        const float* g_v_block = g_v_base + offset;
        // load K V
        #pragma unroll
        for(int idx = tid; idx < HEAD_DIM * BLOCK_N; idx += THREAD_PER_BLOCK) {
            k_smem[idx] = g_k_block[idx];
        }
        #pragma unroll
        for(int idx = tid; idx < HEAD_DIM * BLOCK_N; idx += THREAD_PER_BLOCK) {
            v_smem[idx] = g_v_block[idx];
        }
        __syncthreads();

        // 一个 warp 处理一行 q
        for(int q_i = warp_id; q_i < BLOCK_M; q_i += WARPS) {
            // 遍历 k 的行
            for(int k_j = lane_id; k_j < BLOCK_N; k_j += 32) {
                float local_acc = 0.0f;
                for(int col = 0; col < HEAD_DIM; col++) {
                    local_acc += q_smem[q_i * HEAD_DIM + col] * k_smem[k_j * HEAD_DIM + col];
                }
                // scale
                local_acc *= rsqrtf((float)HEAD_DIM);

                // warp 内 O compute
                float old_max = m_smem[q_i];
                float old_sum = l_smem[q_i];
                // 算 max
                float local_max = local_acc;
                #pragma unroll
                for(int offset = 16; offset >= 1; offset >>= 1) {
                    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                }
                local_max = __shfl_sync(0xffffffff, local_max, 0);

                float new_max = fmaxf(local_max, old_max);
                float exp_sum_factor = expf(old_max - new_max);
                float exp_acc_factor = expf(local_acc - new_max);

                // 算 sum
                float local_sum = exp_acc_factor;
                #pragma unroll
                for(int offset = 16; offset >= 1; offset >>= 1) {
                    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
                }
                local_sum = __shfl_sync(0xffffffff, local_sum, 0);

                float new_sum = local_sum + old_sum * exp_sum_factor;

                if (lane_id == 0) {
                    m_smem[q_i] = new_max;
                    l_smem[q_i] = new_sum;
                }
                __syncwarp();

                // 算 Oi
                int j_base = __shfl_sync(0xffffffff, k_j, 0);
                for(int col = lane_id; col < HEAD_DIM; col += 32) {
                    float new_o_acc = 0.0f;
                    #pragma unroll
                    for(int jj = 0; jj < 32; jj++) {
                        int j = j_base + jj;
                        float exp_acc_factor_reg = __shfl_sync(0xffffffff, exp_acc_factor, jj);
                        new_o_acc += v_smem[j * HEAD_DIM + col] * exp_acc_factor_reg;
                    }

                    o_smem[q_i * HEAD_DIM + col] = 
                        (o_smem[q_i * HEAD_DIM + col] * old_sum * exp_sum_factor + new_o_acc) / new_sum;
                }
            }
        }

        __syncthreads();
    } 

    // write-back
    #pragma unroll
    for(int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += THREAD_PER_BLOCK) {
        g_o_block[idx] = o_smem[idx];
    }
}

void flash_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len
) {
    dim3 gridDim((query_len + BLOCK_M - 1) / BLOCK_M, num_heads, batch_size);
    dim3 blockDim(THREAD_PER_BLOCK);
    kernel_flash_attention<<<gridDim, blockDim>>>(
        q, k, v,
        output,
        batch_size, num_heads, query_len, key_len
    );
    CUDA_CHECK(cudaGetLastError());
}