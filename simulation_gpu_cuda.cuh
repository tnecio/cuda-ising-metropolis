// NOTE: CUDA code must be separated from Boost code
// This file provides an abstraction over all CUDA functionality used
// in GPU simulation

#ifndef ISING_SIMULATION_GPU_CUDA_CUH
#define ISING_SIMULATION_GPU_CUDA_CUH

#include <curand.h>


class CudaPRNG {
    curandGenerator_t gen;
    int size;

public:
    CudaPRNG(unsigned long long int seed, int size_);

    curandStatus generate();

    float *where;
};

struct DeviceMemoryParams {
    int *spins;
    float *interaction;
    float *external_field;
    float beta;
    float magnetic_moment;

    CudaPRNG prng;
    int n;

    int *out_spins;

    DeviceMemoryParams(unsigned long long int seed, int prng_size)
    : prng(seed, prng_size) {
    }
};

void execute_one_step(struct DeviceMemoryParams &dev);

template<typename T>
T *copy_array_to_device(int size, T *from);

template<typename T>
cudaError copy_array_from_device(int size, T *from, T *to);

#endif //ISING_SIMULATION_GPU_CUDA_CUH
