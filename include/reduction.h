void reduction_naive(const float* input, float* output, int N);
void reduction_presum(const float* input, float* output, int N);
void reduction_presum_float4(const float* input, float* output, int N);
void reduction_shuffle(const float* input, float* output, int N);
void reduction_grid_stride(const float* input, float* output, int N);
void reduction_integrate(const float* input, float* output, int N);