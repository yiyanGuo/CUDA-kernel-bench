#pragma once

void rope_naive(float *q, float *k, const float *cos, const float *sin,
                const int *position_ids, int batch_size, int seq_len,
                int num_q_heads, int num_kv_heads, int head_dim,
                int rotary_dim, int position_offset);
