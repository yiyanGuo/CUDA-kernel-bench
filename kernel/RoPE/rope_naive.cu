#include <cuda_runtime_api.h>
#include <cuda_utils.h>
#include <cuda_fp16.h>
#include "rope.h"

#define THREAD_PER_BLOCK 64

__global__ void kernel_rope_naive(
  float* q,
  float* k,
  const float* cos,
  const float* sin,
  const int* position_ids,

  int batch_size,
  int seq_len,
  int num_q_heads,
  int num_kv_heads,
  int head_dim,
  int rotary_dim,
  int position_offset
) {
  int h_all = blockIdx.x;
  int s = blockIdx.y;
  int b = blockIdx.z;

  float* row;
  if (h_all < num_q_heads) {
    int h = h_all;
    row = q + (((b * seq_len + s) * num_q_heads + h) * head_dim);
  } else {
    int h = h_all - num_q_heads;
    row = k + (((b * seq_len + s) * num_kv_heads + h) * head_dim);
  }

  int pos;
  if(position_ids != nullptr) {
    pos = position_ids[blockIdx.z * seq_len + blockIdx.y] + position_offset;
  } else {
    pos = position_offset + blockIdx.y;
  }

  const int tid = threadIdx.x;
  const int pairs = rotary_dim / 2;
  for(int pid = tid; pid < pairs; pid += THREAD_PER_BLOCK) {
    float cos_val = cos[pos * pairs + pid];
    float sin_val = sin[pos * pairs + pid];
    float2 val = *reinterpret_cast<float2*>(&row[2 * pid]);
    float r_x = val.x * cos_val - val.y * sin_val;
    float r_y = val.y * cos_val + val.x * sin_val;
    float2 res = make_float2(r_x, r_y);
    *reinterpret_cast<float2*>(&row[2 * pid]) = res; 
  }
}

void rope_naive(
  float* q,
  float* k,
  const float* cos,
  const float* sin,
  const int* position_ids,

  int batch_size,
  int seq_len,
  int num_q_heads,
  int num_kv_heads,
  int head_dim,
  int rotary_dim,
  int position_offset
) {
  dim3 gridDim((num_q_heads + num_kv_heads), seq_len, batch_size);
  dim3 blockDim = THREAD_PER_BLOCK;

  kernel_rope_naive<<<gridDim, blockDim>>>(
    q, k, cos, sin, 
    position_ids, batch_size, seq_len, num_q_heads, num_kv_heads,
    head_dim, rotary_dim, position_offset
  );
  CUDA_CHECK(cudaGetLastError());
  
}
