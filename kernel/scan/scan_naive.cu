#include "cuda_runtime.h"
#include "scan.h"


__global__ void kernel_scan_naive(const float* input, float* output, int N) {
    float runsum = 0.0f;
    for(int i = 0; i < N - 1; i++) {
        runsum += input[i];
        output[i+1] = runsum;
    }
    output[0] = 0;
}

void scan_naive(const float* input, float* output, int N) {
    kernel_scan_naive<<<1, 1>>>(input, output, N);
}