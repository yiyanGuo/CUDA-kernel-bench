#define THREAD_PER_BLOCK 128

void vector_add_naive(const float* A, const float* B, float* C, int N);
void vector_add_float4(const float* A, const float* B, float* C, int N);
