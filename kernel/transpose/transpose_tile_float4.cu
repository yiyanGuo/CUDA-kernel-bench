#include <cuda_runtime.h>

#define BLOCK_M 8
#define BLOCK_N 32

#define TILE_M 64
#define TILE_N 128

#define ELE_PER_WARP TILE_M * TILE_N / BLOCK_M

__global__ void kernel_transpose_tile_float4(const float* input, float* output, int M, int N) {
    __shared__ float buffer[TILE_N][TILE_M];

    const int block_col_base = blockDim.x * blockIdx.x * TILE_N;
    const int block_row_base = blockDim.y * blockIdx.y * TILE_M;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x; // lane 刚好铺满 TILE_N

    const int g_col = block_col_base + lane_id * 4;
    if(g_col > N) {
        return;
    }

    for(int row_offset = warp_id; row_offset < TILE_M; row_offset += BLOCK_M) {
        int g_row = block_row_base + row_offset;
        float4 in = reinterpret_cast<const float4*>(input)[g_row * N + g_col];
        
        int s_row = lane_id * 4;
        int s_col = row_offset;
        buffer[s_row][s_col] = in.x;
        buffer[s_row+1][s_col] = in.y;
        buffer[s_row+2][s_col] = in.z;
        buffer[s_row+3][s_col] = in.w;
    }

    const int write_back_row_base = (TILE_N / BLOCK_M) * warp_id;
    for(int i = lane_id * 4; i < ELE_PER_WARP; i += 32 * 4) {
        int row = i / TILE_M;
        int col = i % TILE_M;
        re
    }
}

void transpose_tile_float4(const float* input, float* output, int M, int N) {
    dim3 blockDim(BLOCK_M, BLOCK_N);
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    kernel_transpose_tile_float4<<<gridDim, blockDim>>>(input, output, M, N);
}