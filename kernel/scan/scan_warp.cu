#include "cuda_runtime.h"
#include "cuda_utils.h"

#define THREAD_PER_BLOCK 256
#define WARPS (THREAD_PER_BLOCK / 32)
#define ELE_PER_THREAD 4
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

static __global__ void kernel_scan_warp(const float *input, float *output, int N,
                                 float *block_sum) {
  __shared__ float warp_sums[WARPS];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = gid * ELE_PER_THREAD;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;

  float4 in = load_float4(input, idx, N);
  float local0 = 0.0f;
  float local1 = in.x;
  float local2 = in.x + in.y;
  float local3 = in.x + in.y + in.z;
  float thread_sum = in.x + in.y + in.z + in.w;

  // warp-scan
  unsigned int mask = 0xffffffff;
  for (int offset = 1; offset < 32; offset <<= 1) {
    float add = __shfl_up_sync(mask, thread_sum, offset);
    if (lane_id >= offset) {
      thread_sum += add;
    }
  }
  // add offset in warp
  float add = __shfl_up_sync(mask, thread_sum, 1);
  if (lane_id > 0) {
    local0 += add;
    local1 += add;
    local2 += add;
    local3 += add;
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
  float add_warp_sums = 0.0f;
  if (warp_id > 0) {
    add_warp_sums = warp_sums[warp_id - 1];
  }

  if (idx + 0 < N)
    output[idx + 0] = add_warp_sums + local0;
  if (idx + 1 < N)
    output[idx + 1] = add_warp_sums + local1;
  if (idx + 2 < N)
    output[idx + 2] = add_warp_sums + local2;
  if (idx + 3 < N)
    output[idx + 3] = add_warp_sums + local3;

  if (tid == blockDim.x - 1) {
    block_sum[blockIdx.x] = warp_sums[WARPS - 1];
  }
}

static __global__ void kernel_add_offset_warp(float *data, float *offset, int N) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int idx = tid * ELE_PER_THREAD;
  const int bid = blockIdx.x;
  float4 old = load_float4(data, idx, N);
  float add = 0.0f;
  add = offset[bid];

  if (idx + 0 < N)
    data[idx + 0] = add + old.x;
  if (idx + 1 < N)
    data[idx + 1] = add + old.y;
  if (idx + 2 < N)
    data[idx + 2] = add + old.z;
  if (idx + 3 < N)
    data[idx + 3] = add + old.w;
}

void scan_warp(const float *input, float *output, int N) {
  if (N <= 0) {
    return;
  }

  int threads = THREAD_PER_BLOCK;
  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&block_sums, sizeof(float) * blocks));

  kernel_scan_warp<<<blocks, threads>>>(input, output, N, block_sums);
  CUDA_CHECK(cudaGetLastError());

  if (blocks > 1) {
    scan_warp(block_sums, block_sums, blocks);

    kernel_add_offset_warp<<<blocks, threads>>>(output, block_sums, N);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaFree(block_sums));
}