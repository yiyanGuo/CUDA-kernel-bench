#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <cstddef>

#define THREAD_PER_BLOCK 256
#define ELE_PER_THREAD 8
#define VEC_PER_THREAD (ELE_PER_THREAD / 4)
#define WARPS (THREAD_PER_BLOCK / 32)
#define BLOCK_SIZE (ELE_PER_THREAD * THREAD_PER_BLOCK)

__device__ __forceinline__ float4 load_float4(const float *input, int idx,
                                              int N) {
  float4 v;

  if (idx + 3 < N) {
    v = *reinterpret_cast<const float4 *>(input + idx);
  } else {
    v.x = (idx + 0 < N) ? input[idx + 0] : 0.0f;
    v.y = (idx + 1 < N) ? input[idx + 1] : 0.0f;
    v.z = (idx + 2 < N) ? input[idx + 2] : 0.0f;
    v.w = (idx + 3 < N) ? input[idx + 3] : 0.0f;
  }

  return v;
}

__device__ __forceinline__ void store_float4(float* output, int idx, int N, float4 out) {
  if(idx + 3 < N) {
    *reinterpret_cast<float4*>(&output[idx]) = out;
  } else {
    if(idx + 0 < N) output[idx + 0] = out.x;
    if(idx + 1 < N) output[idx + 1] = out.y;
    if(idx + 2 < N) output[idx + 2] = out.z;
    if(idx + 3 < N) output[idx + 3] = out.w; 
  }
}

static __global__ void kernel_scan_in_block(const float *input, float* output, int N, float* block_sum) {
  __shared__ float warp_sums[WARPS];
  float local[ELE_PER_THREAD];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = gid * ELE_PER_THREAD;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;


  float thread_sum = 0.0f;

#pragma unroll
  for (int v = 0; v < VEC_PER_THREAD; ++v) {
    float4 in = load_float4(input, idx + v * 4, N);

    local[v * 4 + 0] = thread_sum;
    thread_sum += in.x;

    local[v * 4 + 1] = thread_sum;
    thread_sum += in.y;

    local[v * 4 + 2] = thread_sum;
    thread_sum += in.z;

    local[v * 4 + 3] = thread_sum;
    thread_sum += in.w;
  }

  // warp-scan
  unsigned int mask = 0xffffffff;
  for (int offset = 1; offset < 32; offset <<= 1) {
    float add = __shfl_up_sync(mask, thread_sum, offset);
    if (lane_id >= offset) {
      thread_sum += add;
    }
  }
  // add offset in warp
  float thread_offset = __shfl_up_sync(mask, thread_sum, 1);

  if (lane_id > 0) {
#pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; ++i) {
      local[i] += thread_offset;
    }
  }
  if (lane_id == 31) {
    warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  // scan on warp_sums in one warp
  if (warp_id == 0) {
    float x = (lane_id < WARPS) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 1; offset < WARPS; offset <<= 1) {
      float add = __shfl_up_sync(mask, x, offset);
      if (lane_id >= offset) {
        x += add;
      }
    }
    if(lane_id < WARPS)
        warp_sums[lane_id] = x;
  }
  __syncthreads();

  // add offset of warp_sums
  float warp_offset = 0.0f;
  if (warp_id > 0) {
    warp_offset = warp_sums[warp_id - 1];
  }

#pragma unroll
  for (int v = 0; v < VEC_PER_THREAD; ++v) {
    float4 out = make_float4(
        warp_offset + local[v * 4 + 0],
        warp_offset + local[v * 4 + 1],
        warp_offset + local[v * 4 + 2],
        warp_offset + local[v * 4 + 3]
    );

    store_float4(output, idx + v * 4, N, out);
  }

  if (tid == blockDim.x - 1) {
    block_sum[blockIdx.x] = warp_sums[WARPS - 1];
  }
}

static __global__ void kernel_add_offset(float *data, const float *offset, int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x * ELE_PER_THREAD;

  float add = offset[blockIdx.x];

#pragma unroll
  for (int v = 0; v < VEC_PER_THREAD; ++v) {
    float4 old = load_float4(data, idx + v * 4, N);

    old.x += add;
    old.y += add;
    old.z += add;
    old.w += add;

    store_float4(data, idx + v * 4, N, old);
  }
}


static void scan_helper(const float* input, float* output, int N, float* buffer) {
    if(N <= 0) {
        return;
    }

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_scan_in_block<<<blocks, THREAD_PER_BLOCK>>>(input, output, N, buffer);
    CUDA_CHECK(cudaGetLastError());

    if(blocks > 1) {
        scan_helper(buffer, buffer, blocks, buffer + blocks);

        kernel_add_offset<<<blocks, THREAD_PER_BLOCK>>>(output, buffer, N);
        CUDA_CHECK(cudaGetLastError());
    }
}

void scan_memory_buffer(const float* input, float* output, int N) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* buffer = nullptr;
    float* ptr;
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(float) * blocks * 2));
    ptr = buffer;

    scan_helper(input, output, N, ptr);

    CUDA_CHECK(cudaFree((void*)buffer));
}