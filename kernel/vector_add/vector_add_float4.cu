#include <cuda_runtime.h>
#include "vector_add.h"

__global__ void kernel_vector_add_float4(const float* A, const float* B, float* C, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = tid * 4;
    if(i < N) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        reinterpret_cast<float4*>(C)[i] = c;
    }
}

void vector_add_float4(const float* A, const float* B, float* C, int N) {
    dim3 gridDim((N + 4*THREAD_PER_BLOCK - 1) / 4*THREAD_PER_BLOCK);
    kernel_vector_add_float4<<<THREAD_PER_BLOCK, gridDim>>>(A, B, C, N);
}