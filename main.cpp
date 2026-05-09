#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

void vector_add(
    half* z, int num, const half* x, const half* y, half a, half b, half c
);

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                              \
        }                                                                      \
    } while (0)

int main() {
    constexpr int num = 1 << 20;
    constexpr float a = 1.25f;
    constexpr float b = -0.5f;
    constexpr float c = 2.0f;

    std::vector<half> h_x(num);
    std::vector<half> h_y(num);
    std::vector<half> h_z(num);

    for (int i = 0; i < num; ++i) {
        float x = static_cast<float>(i % 97) * 0.125f;
        float y = static_cast<float>(i % 53) * -0.25f;
        h_x[i] = __float2half(x);
        h_y[i] = __float2half(y);
    }

    half* d_x = nullptr;
    half* d_y = nullptr;
    half* d_z = nullptr;
    const size_t bytes = static_cast<size_t>(num) * sizeof(half);

    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y, bytes));
    CHECK_CUDA(cudaMalloc(&d_z, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    vector_add(d_z, num, d_x, d_y, __float2half(a), __float2half(b), __float2half(c));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_abs_error = 0.0f;
    for (int i = 0; i < num; ++i) {
        float x = __half2float(h_x[i]);
        float y = __half2float(h_y[i]);
        float actual = __half2float(h_z[i]);
        float expected = __half2float(__float2half(a * x + b * y + c));
        float abs_error = std::abs(actual - expected);
        max_abs_error = std::max(max_abs_error, abs_error);

        if (abs_error > 1e-3f) {
            if (errors < 10) {
                std::cerr << "Mismatch at " << i << ": got " << actual
                          << ", expected " << expected << std::endl;
            }
            ++errors;
        }
    }

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_z));

    if (errors != 0) {
        std::cerr << "FAILED: " << errors << " mismatches, max abs error "
                  << max_abs_error << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "PASSED: CUTE/CUTLASS include path works and vector_add result is correct. "
              << "max abs error = " << max_abs_error << std::endl;
    return EXIT_SUCCESS;
}
