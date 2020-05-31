
#ifndef ISING_TEST_SIMULATION_GPU_CUDA_H
#define ISING_TEST_SIMULATION_GPU_CUDA_H

#include "utils.h"

#include "../simulation_cpu.h"

#include "../simulation_gpu_cuda.cuh"

int test_cuda_prng() {
    CudaPRNG cuda_prng(2019ULL, 1024);
    curandStatus cs = cuda_prng.generate();
    if (cs) {
        std::ostringstream msg;
        msg << cs;
        fail("CudaPRNG", msg.str());
    }
    float arr[1024];
    copy_array_from_device(1024, cuda_prng.where, arr);

    CudaPRNG cuda_prng2(2019ULL, 1024);
    cuda_prng2.generate();
    float arr2[1024];
    copy_array_from_device(1024, cuda_prng2.where, arr2);

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
    float *devmem = copy_array_to_device(10, src);
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

//template<typename InType, typename OutType>
//OutType *vector_to_array(const vector<InType> &in) {
//    OutType *out = new OutType[in.size()];
//    for (size_t i = 0; i < in.size(); i++) {
//        out[i] = (OutType) in(i);
//    }
//    return out;
//}
//
//
//int *copy_bool_vector_to_gpu(const vector<bool> &v) {
//    int *host_arr = vector_to_array<bool, int>(v);
//    int *dev_arr = copy_array_to_device<int>(v.size(), host_arr);
//    delete[] host_arr;
//    return dev_arr;
//}
//
//float *matrix_to_float_array(const matrix<double> &in) {
//    float *out = new float[in.size1() * in.size2()];
//    for (size_t i = 0; i < in.size1(); i++) {
//        for (size_t j = 0; j < in.size2(); j++) {
//            out[i * in.size2() + j] = (float) in(i, j);
//        }
//    }
//    return out;
//}
//
//int test_execute_one_step() {
//    GeneralisedIsingParams params = prepare_params(10);
//    struct DeviceMemoryParams dev(2020ULL, params.initial_spins.size());
//    dev.spins = copy_bool_vector_to_gpu(params.initial_spins);
//    dev.out_spins = copy_bool_vector_to_gpu(params.initial_spins);
//    dev.interaction = copy_matrix_to_gpu(params.interaction);
//    dev.external_field = copy_float_vector_to_gpu(params.external_field);
//    dev.n = params.initial_spins.size();
//    dev.beta = 1. / (BOLTZMANN * params.temperature);
//    dev.magnetic_moment = params.magnetic_moment;
//    // TODO: clean-up the above copy-paste
//
//    // Idea: run one step on each device and compare
//    CPUGeneralisedIsingModel cpu(params);
//    GPUGeneralisedIsingModel gpu(params);
//
//    cpu.run(1);
//    vector<bool> expected = cpu.get_spins();
//    gpu.run(1);
//    vector<bool> actual = gpu.get_spins();
//
//    print_spins(expected);
//    print_spins(actual);
//
//    for (int i = 0; i < params.initial_spins.size(); i++) {
//        if (expected(i) != actual(i)) {
//            return fail("execute_one_step", "Booyah");
//        }
//    }
//
//    return success("execute_one_step");
//}

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
    cudaError ce;
    ce = copy_array_from_device(10, devmem, dest);
    if (ce) {
        std::ostringstream os;
        os << "cudaMemcpy back " << ce;
        return fail(name, os.str());
    }
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
//    TEST(test_execute_one_step);
    TEST(test_copy_array_from_device);
TEST_SUITE_END

#endif //ISING_TEST_SIMULATION_GPU_CUDA_H
