#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Shared memory per SM: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("L2 cache size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}
