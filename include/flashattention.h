#pragma once

#include <cuda_fp16.h>

void flash_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len);

void flash_attention_mma(
    const half* q,
    const half* k,
    const half* v,
    half* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len);
