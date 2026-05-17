#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>

#define THREAD_PER_BLOCK 128
#define WARPS (THREAD_PER_BLOCK / 32)
#define BLOCK_M 64
#define BLOCK_N 64
#define HEAD_DIM 64

__device__ __forceinline__ uint32_t pack_f32_to_f16x2(float lo, float hi) {
    uint32_t r;
    asm volatile(
        "cvt.rn.f16x2.f32 %0, %1, %2;"
        : "=r"(r)
        : "f"(hi), "f"(lo)
    );
    return r;
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 uaddr; cvta.to.shared.u64 uaddr, %1; cvt.u32.u64 %0, uaddr; }\n"
        : "=r"(addr) : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(uint32_t a[4], uint32_t b[2], float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

__device__ __forceinline__ void copy_half_tile_16b(half* dst, const half* src, int num_half) {
    constexpr int HALF_PER_VEC = sizeof(int4) / sizeof(half);
    int vec_count = num_half / HALF_PER_VEC;
    int4* dst_vec = reinterpret_cast<int4*>(dst);
    const int4* src_vec = reinterpret_cast<const int4*>(src);
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
        dst_vec[idx] = src_vec[idx];
    }
}

__global__ void kernel_flash_attention_integrated(
    const half* q,
    const half* k,
    const half* v,
    half* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len
) {
    __shared__ __align__(16) half q_smem[BLOCK_M * HEAD_DIM];
    __shared__ __align__(16) half k_smem[BLOCK_N * HEAD_DIM];
    __shared__ __align__(16) half v_smem[BLOCK_N * HEAD_DIM];

    float scale = rsqrtf((float)HEAD_DIM);

    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int q_block_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int q_i = warp_id * 16;
    const half* g_k_base = k + batch * (num_heads * key_len * HEAD_DIM) + head * (key_len * HEAD_DIM);
    const half* g_v_base = v + batch * (num_heads * key_len * HEAD_DIM) + head * (key_len * HEAD_DIM);
    const half* g_q_block = q + batch * (num_heads * query_len * HEAD_DIM) + head * (query_len * HEAD_DIM) + q_block_id * BLOCK_M * HEAD_DIM;
    half* g_o_block = output + batch * (num_heads * query_len * HEAD_DIM) + head * (query_len * HEAD_DIM) + q_block_id * BLOCK_M * HEAD_DIM;

    copy_half_tile_16b(q_smem, g_q_block, BLOCK_M * HEAD_DIM);
    __syncthreads();

    int group_id = lane_id / 4;
    int tid4 = lane_id % 4;
    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float o_frag[HEAD_DIM / 8][4];

    #pragma unroll
    for (int vc = 0; vc < HEAD_DIM / 8; ++vc) {
        o_frag[vc][0] = 0.0f;
        o_frag[vc][1] = 0.0f;
        o_frag[vc][2] = 0.0f;
        o_frag[vc][3] = 0.0f;
    }

    for(int offset = 0; offset < key_len * HEAD_DIM; offset += BLOCK_N * HEAD_DIM) {
        const half* g_k_block = g_k_base + offset;
        const half* g_v_block = g_v_base + offset;
        copy_half_tile_16b(k_smem, g_k_block, BLOCK_N * HEAD_DIM);
        copy_half_tile_16b(v_smem, g_v_block, BLOCK_N * HEAD_DIM);
        __syncthreads();

        constexpr int NUM_S_ACC = BLOCK_N / 8;
        float s_acc[NUM_S_ACC][4];
        #pragma unroll
        for (int ns = 0; ns < NUM_S_ACC; ++ns) {
            s_acc[ns][0] = 0.f;
            s_acc[ns][1] = 0.f;
            s_acc[ns][2] = 0.f;
            s_acc[ns][3] = 0.f;
        }

        #pragma unroll
        for(int d0 = 0; d0 < HEAD_DIM; d0 += 16) {
            uint32_t a[4];
            int a_mat  = lane_id >> 3;
            int a_row8 = lane_id & 7;

            int a_row = q_i + a_row8 + ((a_mat & 1) ? 8 : 0);
            int a_col = d0 + ((a_mat >= 2) ? 8 : 0);

            uint32_t a_addr = smem_u32(q_smem + a_row * HEAD_DIM + a_col);
            ldmatrix_x4(a, a_addr);

            #pragma unroll
            for(int ns = 0; ns < NUM_S_ACC; ns++) {
                uint32_t b[2];
                int b_mat   = (lane_id >> 3) & 1;
                int b_row_n = ns * 8 + (lane_id & 7);
                int b_col_d = d0 + b_mat * 8;
                uint32_t b_addr = smem_u32(k_smem + b_row_n * HEAD_DIM + b_col_d);
                ldmatrix_x2(b, b_addr);

                mma_m16n8k16(a, b, s_acc[ns]);
            }
        }

        #pragma unroll
        for(int ns = 0; ns < NUM_S_ACC; ns++) {
            s_acc[ns][0] *= scale;
            s_acc[ns][1] *= scale;
            s_acc[ns][2] *= scale;
            s_acc[ns][3] *= scale;
        }

        float local_max0 = -INFINITY;
        float local_max1 = -INFINITY;
        #pragma unroll
        for(int ns = 0; ns < NUM_S_ACC; ns++) {
            local_max0 = fmaxf(local_max0, s_acc[ns][0]);
            local_max0 = fmaxf(local_max0, s_acc[ns][1]);

            local_max1 = fmaxf(local_max1, s_acc[ns][2]);
            local_max1 = fmaxf(local_max1, s_acc[ns][3]);
        }
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 1, 4));
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 2, 4));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 1, 4));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 2, 4));

        float old_max0 = m0;
        float old_max1 = m1;
        local_max0 = fmaxf(local_max0, old_max0);
        local_max1 = fmaxf(local_max1, old_max1);
        m0 = local_max0;
        m1 = local_max1;

        float local_sum0 = 0.0f;
        float local_sum1 = 0.0f;
        #pragma unroll
        for(int ns = 0; ns < NUM_S_ACC; ns++){
            float s_acc_ns_0 = __expf(s_acc[ns][0] - local_max0);
            float s_acc_ns_1 = __expf(s_acc[ns][1] - local_max0);
            float s_acc_ns_2 = __expf(s_acc[ns][2] - local_max1);
            float s_acc_ns_3 = __expf(s_acc[ns][3] - local_max1);
            s_acc[ns][0] = s_acc_ns_0;
            s_acc[ns][1] = s_acc_ns_1;
            s_acc[ns][2] = s_acc_ns_2;
            s_acc[ns][3] = s_acc_ns_3;
            local_sum0 += s_acc_ns_0 + s_acc_ns_1;
            local_sum1 += s_acc_ns_2 + s_acc_ns_3;
        }
        local_sum0 += __shfl_xor_sync(0xffffffff, local_sum0, 1, 4);
        local_sum0 += __shfl_xor_sync(0xffffffff, local_sum0, 2, 4);
        local_sum1 += __shfl_xor_sync(0xffffffff, local_sum1, 1, 4);
        local_sum1 += __shfl_xor_sync(0xffffffff, local_sum1, 2, 4);

        float old_scale0 = __expf(old_max0 - local_max0);
        float old_scale1 = __expf(old_max1 - local_max1);
        local_sum0 += l0 * old_scale0;
        local_sum1 += l1 * old_scale1;
        l0 = local_sum0;
        l1 = local_sum1;

        #pragma unroll
        for(int vc = 0; vc < HEAD_DIM / 8; ++vc) {
            int v_col = vc * 8;
            o_frag[vc][0] *= old_scale0;
            o_frag[vc][1] *= old_scale0;
            o_frag[vc][2] *= old_scale1;
            o_frag[vc][3] *= old_scale1;

            #pragma unroll
            for(int mma_block = 0; mma_block < BLOCK_N / 16; mma_block++) {
                uint32_t p[4];
                p[0] = pack_f32_to_f16x2(s_acc[2 * mma_block][0], s_acc[2 * mma_block][1]);
                p[1] = pack_f32_to_f16x2(s_acc[2 * mma_block][2], s_acc[2 * mma_block][3]);
                p[2] = pack_f32_to_f16x2(s_acc[2 * mma_block + 1][0], s_acc[2 * mma_block + 1][1]);
                p[3] = pack_f32_to_f16x2(s_acc[2 * mma_block + 1][2], s_acc[2 * mma_block + 1][3]);

                uint32_t v_frag[2];
                int ldm_lane = lane_id & 15;
                int v_mat    = ldm_lane >> 3;
                int v_row    = mma_block * 16 + v_mat * 8 + (ldm_lane & 7);

                uint32_t v_addr = smem_u32(v_smem + v_row * HEAD_DIM + v_col);
                ldmatrix_x2_trans(v_frag, v_addr);

                mma_m16n8k16(p, v_frag, o_frag[vc]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int vc = 0; vc < HEAD_DIM / 8; ++vc) {
        int v_col = vc * 8;
        g_o_block[(q_i + group_id) * HEAD_DIM + v_col + tid4 * 2 + 0] = __float2half(o_frag[vc][0] / l0);
        g_o_block[(q_i + group_id) * HEAD_DIM + v_col + tid4 * 2 + 1] = __float2half(o_frag[vc][1] / l0);
        g_o_block[(q_i + group_id + 8) * HEAD_DIM + v_col + tid4 * 2 + 0] = __float2half(o_frag[vc][2] / l1);
        g_o_block[(q_i + group_id + 8) * HEAD_DIM + v_col + tid4 * 2 + 1] = __float2half(o_frag[vc][3] / l1);
    }
}

void flash_attention_integrated(
    const half* q,
    const half* k,
    const half* v,
    half* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len
) {
    dim3 gridDim((query_len + BLOCK_M - 1) / BLOCK_M, num_heads, batch_size);
    dim3 blockDim(THREAD_PER_BLOCK);
    kernel_flash_attention_integrated<<<gridDim, blockDim>>>(q, k, v, output, batch_size, num_heads, query_len, key_len);
    CUDA_CHECK(cudaGetLastError());
}
