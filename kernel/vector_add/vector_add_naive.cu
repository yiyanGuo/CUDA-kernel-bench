#include <cuda_utils.h>

__global__ void kernel_vector_add_naive(const float *a, const float *b,
                                        float *c, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

void vector_add_naive(const float *a, const float *b, float *c, int n) {
  constexpr int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  kernel_vector_add_naive<<<grid_size, block_size>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());
}