#pragma once

#include <cuda_fp16.h>

#include "cuda_utils.h"

void rms_naive(const float *input, const float *weight, float *output,
               int num_tokens, int hidden_size, float eps);

void rms_naive_v2(const float *input, const float *weight, float *output,
                  int num_tokens, int hidden_size, float eps);

void rms_shared_memory(const float *input, const float *weight, float *output,
                       int num_tokens, int hidden_size, float eps);

void rms_half(const half *input, const half *weight, half *output,
              int num_tokens, int hidden_size, float eps);

void rms_half2(const half *input, const half *weight, half *output,
               int num_tokens, int hidden_size, float eps);
