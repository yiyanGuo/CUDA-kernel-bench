#include <cuda_utils.h>

__global__ void kernel_transpose_naive(const float *input, float *output,
                                       int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    output[col * rows + row] = input[row * cols + col];
  }
}

void transpose_naive(const float *input, float *output, int rows, int cols) {
  dim3 block_size(16, 16);
  dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                 (rows + block_size.y - 1) / block_size.y);
  kernel_transpose_naive<<<grid_size, block_size>>>(input, output, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}
