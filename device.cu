/*
#
# device.cu
#
# Poll available GPUs and print some information.
#
# sheneman@uidaho.edu
# 2024
#
#
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, i);

        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);

        // Calculate total number of warps per multiprocessor:
        int warpsPerMultiprocessor = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        printf("  Warps per multiprocessor: %d\n", warpsPerMultiprocessor);

        // Calculate total potential warps for the device:
        int totalWarps = warpsPerMultiprocessor * prop.multiProcessorCount;
        printf("  Total potential warps: %d\n", totalWarps);
    }
    return 0;
}

