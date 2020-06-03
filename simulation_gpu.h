#ifndef ISING_SIMULATION_GPU_H
#define ISING_SIMULATION_GPU_H

#include <iostream> // TODO separete to failure module

#include "simulation.h"

#include <curand.h>

void cuda_failure(std::string stmt, int ce);

#define CUDA_HANDLERR(stmt) \
do { \
    int ce = stmt; \
    if (ce) { \
        cuda_failure(#stmt, ce); \
    } \
} while (false)

class CudaPRNG {
    curandGenerator_t gen;
    int size;

public:
    CudaPRNG(unsigned long long int seed, int size_);

    void generate();

    float *where;
};


class CudaDevice {
public:
    template<typename DevType, typename HostType>
    static DevType *copy_array_to_device_with_conversion(
            int size, const HostType *from);

    template<typename T>
    static T *copy_array_to_device(int size, const T *from);

    template<typename DevType, typename HostType>
    static void copy_array_from_device_with_conversion(
            int size, const DevType *from, HostType *to);

    template<typename T>
    static void copy_array_from_device(int size, const T *from, T *to);

    static void free_array_from_device(void *arr);
};


struct IsingParamsDevice {
    int *spins;
    int n;

    float beta;
    float magnetic_moment;
};

struct Simple2DIsingParamsDevice : public IsingParamsDevice {
    int xlen;
    int ylen;

    float interaction;
    float external_field;

    Simple2DIsingParamsDevice(const Simple2DIsingParams &params);

    ~Simple2DIsingParamsDevice() {
        //CudaDevice::free_array_from_device(spins);
        // TODO: this gets called as many times as this object is copied
        // but the memory is initialised only once, when it is created
        // through the constructor from CPU params
    }
};


class GPUModel {
protected:
    CudaPRNG prng;

public:
    GPUModel(unsigned long long int seed, IsingParams &params)
            : prng(seed, params.initial_spins.size()) {
    }
};

class GPUSimple2DIsingModel : public Simple2DIsingModel, public GPUModel {
protected:
    const Simple2DIsingParamsDevice dev;

    void execute_simulation_step_on_gpu();

    void read_spins_from_gpu();

public:
    GPUSimple2DIsingModel(Simple2DIsingParams initial_params)
            : Simple2DIsingModel(initial_params),
              GPUModel(2020ULL, initial_params),
              dev(initial_params) {
    }

    void run(size_t max_steps);
};

#endif //ISING_SIMULATION_GPU_H
