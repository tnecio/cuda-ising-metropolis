
#ifndef ISING_TEST_SIMULATION_GPU_CUDA_H
#define ISING_TEST_SIMULATION_GPU_CUDA_H

#include <string>
#include <iostream>
#include <sstream>

#include "utils.h"

#include "../simulation_cpu.h"

#include "../simulation_gpu.h"

int test_cuda_prng() {
    CudaPRNG cuda_prng(2019ULL, 1024);
    cuda_prng.generate();
    float arr[1024];
    CudaDevice::copy_array_from_device(1024, cuda_prng.where, arr);

    CudaPRNG cuda_prng2(2019ULL, 1024);
    cuda_prng2.generate();
    float arr2[1024];
    CudaDevice::copy_array_from_device(1024, cuda_prng2.where, arr2);

    for (int i = 0; i < 1024; i++) {
        if (std::abs(arr[i] - arr2[i]) > 0.001 ||
            arr[i] < 0 || arr[i] > 1) {
            std::ostringstream msg;
            msg << "i: " << i << " arr: " << arr[i] << " arr2: " << arr2[i];
            return fail("CudaPRNG", msg.str());
        }
    }

    return success("CudaPRNG");
}

int test_copy_array_to_device() {
    std::string name = "copy_array_to_device";
    float src[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    float *devmem = CudaDevice::copy_array_to_device(10, src);
    float dest[10];
    cudaError ce = cudaMemcpy(dest, devmem, 10 * sizeof(float),
                              cudaMemcpyDeviceToHost);
    if (ce) {
        return fail(name, "cudaMemcpy back");
    }
    for (int i = 0; i < 10; i++) {
        if (dest[i] != src[i]) {
            return fail(name, "dest != src");
        }
    }
    return success(name);
}

int test_copy_array_from_device() {
    std::string name = "copy_array_from_device";
    float *devmem;
    if (cudaMalloc(&devmem, 10 * sizeof(float))) {
        return fail(name, "cudaMalloc");
    }
    float src[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    if (cudaMemcpy(devmem, src, 10 * sizeof(float), cudaMemcpyHostToDevice)) {
        return fail(name, "cudaMemcpy");
    }
    float dest[10];
    CudaDevice::copy_array_from_device(10, devmem, dest);
    for (int i = 0; i < 10; i++) {
        if (dest[i] != src[i]) {
            std::cout << "Dest: " << dest[i] << " Src: " << src[i] << std::endl;
            return fail(name, "dest != src");
        }
    }
    return success(name);
}

TEST_SUITE_START(test_gpu_cuda_functions)
    TEST(test_cuda_prng);
    TEST(test_copy_array_to_device);
    TEST(test_copy_array_from_device);
TEST_SUITE_END

#endif //ISING_TEST_SIMULATION_GPU_CUDA_H
