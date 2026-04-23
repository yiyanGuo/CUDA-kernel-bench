#include <cuda_runtime.h>

#define TILE_DIM 32

#define BLOCK_M 4
#define BLOCK_N 32

#define THREAD_STRIDE 8

__global__ void kernel_transpose_tile(const float* input, float* output, int M, int N) {
    __shared__ float buffer[TILE_DIM][TILE_DIM];

    const int block_g_input_row = blockIdx.y * TILE_DIM;
    const int block_g_input_col = blockIdx.x * TILE_DIM;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    for(int s_row = warp_id; s_row < TILE_DIM; s_row += THREAD_STRIDE) {
        int row = s_row + block_g_input_row;
        int col = lane_id + block_g_input_col;
        buffer[lane_id][s_row] = input[row * N + col];
    }

    __syncthreads();

    const int block_g_output_row = block_g_input_col;
    const int block_g_output_col = block_g_input_row;

    for(int s_row = warp_id; s_row < TILE_DIM; s_row += THREAD_STRIDE) {
        int row = s_row + block_g_output_row;
        int col = lane_id + block_g_output_col;
        output[row * M + col] = buffer[s_row][lane_id];
    }
}

void transpose_tile(const float* input, float* output, int M, int N) {
    dim3 blockDim(BLOCK_M, BLOCK_N);
    dim3 gridDim((M + TILE_DIM -1)/ TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    kernel_transpose_tile<<<gridDim, blockDim>>>(input, output, M, N);
}