void scan_naive(const float* input, float* output, int N);
void scan_one_block(const float* input, float* output, int N);
void scan_multi_block(const float* input, float* output, int N);
void scan_thrust_exclusive(const float* input, float* output, int N);
void scan_warp(const float* input, float* output, int N);