#include "cuda_runtime.h"
#include "cuda_utils.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>


void scan_thrust_exclusive(const float* input, float* output, int N) {
    auto in = thrust::device_pointer_cast(input);
    auto out = thrust::device_pointer_cast(output);
    thrust::exclusive_scan(in, in + N, out);
}