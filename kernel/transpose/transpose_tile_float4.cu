#include <cuda_runtime.h>

#include "cuda_utils.h"

#define BLOCK_M 8
#define BLOCK_N 32

#define TILE_M 64
#define TILE_N 128

#define ELE_PER_WARP (TILE_M * TILE_N / BLOCK_M)

__global__ void kernel_transpose_tile_float4(const float* input, float* output, int M, int N) {
    __shared__ float buffer[TILE_N][TILE_M];

    const int block_col_base = blockIdx.x * TILE_N;
    const int block_row_base = blockIdx.y * TILE_M;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    const int g_col = block_col_base + lane_id * 4;

    for(int row_offset = warp_id; row_offset < TILE_M; row_offset += BLOCK_M) {
        int g_row = block_row_base + row_offset;
        float4 in{0.0f, 0.0f, 0.0f, 0.0f};

        if (g_row < M && g_col < N) {
            if (g_col + 3 < N) {
                in = *reinterpret_cast<const float4*>(input + g_row * N + g_col);
            } else {
                in.x = input[g_row * N + g_col + 0];
                in.y = (g_col + 1 < N) ? input[g_row * N + g_col + 1] : 0.0f;
                in.z = (g_col + 2 < N) ? input[g_row * N + g_col + 2] : 0.0f;
                in.w = (g_col + 3 < N) ? input[g_row * N + g_col + 3] : 0.0f;
            }
        }
        
        int s_row = lane_id * 4;
        int s_col = row_offset;
        buffer[s_row][s_col] = in.x;
        buffer[s_row+1][s_col] = in.y;
        buffer[s_row+2][s_col] = in.z;
        buffer[s_row+3][s_col] = in.w;
    }
    __syncthreads();

    const int write_back_row_base = (TILE_N / BLOCK_M) * warp_id;
    for(int i = lane_id * 4; i < ELE_PER_WARP; i += BLOCK_N * 4) {
        int row = i / TILE_M;
        int col = i % TILE_M;
        int out_row = block_col_base + write_back_row_base + row;
        int out_col = block_row_base + col;

        if (out_row < N && out_col < M) {
            output[out_row * M + out_col] =
                buffer[write_back_row_base + row][col];
        }
        if (out_row < N && out_col + 1 < M) {
            output[out_row * M + out_col + 1] =
                buffer[write_back_row_base + row][col + 1];
        }
        if (out_row < N && out_col + 2 < M) {
            output[out_row * M + out_col + 2] =
                buffer[write_back_row_base + row][col + 2];
        }
        if (out_row < N && out_col + 3 < M) {
            output[out_row * M + out_col + 3] =
                buffer[write_back_row_base + row][col + 3];
        }
    }
}

void transpose_tile_float4(const float* input, float* output, int M, int N) {
    if (M <= 0 || N <= 0) {
        return;
    }

    dim3 blockDim(BLOCK_N, BLOCK_M);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    kernel_transpose_tile_float4<<<gridDim, blockDim>>>(input, output, M, N);
    CUDA_CHECK(cudaGetLastError());
}
