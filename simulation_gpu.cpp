
#include "simulation_gpu.h"

void GPUGeneralisedIsingModel::run(uint no_steps) {
    GeneralisedModelDeviceMemoryParams dev = prepare_gpu();
    // TODO: separate preparation from running for speed-measurement purposes
    for (uint i = 0; i < no_steps; i++) {
        execute_simulation_step_on_gpu(dev);
        this->step_count += 1;
    }
    read_spins_from_gpu(dev);
}

void GPUGeneralisedIsingModel::read_spins_from_gpu(
        struct GeneralisedModelDeviceMemoryParams dev) {
    // Note: final spins are in `spins`, not in `out_spins`"
    int *arr = new int[spins.size()];
    copy_array_from_device<int>(dev.n, dev.spins, arr);
    for (size_t i = 0; i < spins.size(); i++) {
        spins(i) = arr[i];
    }
    delete [] arr;
}

void GPUGeneralisedIsingModel::execute_simulation_step_on_gpu(
        struct GeneralisedModelDeviceMemoryParams &dev) const {
    // Execute one step
    execute_one_step(dev);

    // Switch in spins (ptr) with out spins (ptr)
    int *tmp = dev.spins;
    dev.spins = dev.out_spins;
    dev.out_spins = tmp;
}

template<typename InType, typename OutType>
OutType *vector_to_array(const vector<InType> &in) {
    OutType *out = new OutType[in.size()];
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = (OutType) in(i);
    }
    return out;
}

template<typename CpuType, typename GpuType>
GpuType *copy_vector_to_gpu(const vector<CpuType> &v) {
    GpuType *host_arr = vector_to_array<CpuType, GpuType>(v);
    GpuType *dev_arr = copy_array_to_device<GpuType>(v.size(), host_arr);
    delete[] host_arr;
    return dev_arr;
}

float *matrix_to_float_array(const matrix<double> &in) {
    float *out = new float[in.size1() * in.size2()];
    for (size_t i = 0; i < in.size1(); i++) {
        for (size_t j = 0; j < in.size2(); j++) {
            out[i * in.size2() + j] = (float) in(i, j);
        }
    }
    return out;
}

float *copy_matrix_to_gpu(const matrix<double> &m) {
    float *host_arr = matrix_to_float_array(m);
    float *dev_arr =
            copy_array_to_device<float>(m.size1() * m.size2(), host_arr);
    delete[] host_arr;
    return dev_arr;
}

struct GeneralisedModelDeviceMemoryParams
GPUGeneralisedIsingModel::prepare_gpu() const {
    struct GeneralisedModelDeviceMemoryParams dev(2020ULL, spins.size());
    dev.spins = copy_vector_to_gpu<int, int>(spins);
    dev.out_spins = copy_vector_to_gpu<int, int>(spins);
    dev.interaction = copy_matrix_to_gpu(params.interaction);
    dev.external_field = copy_vector_to_gpu<double, float>(
            params.external_field);
    dev.n = spins.size();
    dev.beta = 1. / (BOLTZMANN * params.temperature);
    dev.magnetic_moment = params.magnetic_moment;
    return dev;
}


void GPUSimple2DIsingModel::run(uint no_steps) {
    Simple2DModelDeviceMemoryParams dev = prepare_gpu();
    // TODO: separate preparation from running for speed-measurement purposes
    for (uint i = 0; i < no_steps; i++) {
        execute_simulation_step_on_gpu(dev);
        this->step_count += 1;
    }
    read_spins_from_gpu(dev);
}

void GPUSimple2DIsingModel::read_spins_from_gpu(
        struct Simple2DModelDeviceMemoryParams dev) {
    // TODO: take in "AbstractDeviceMemoryParams"
    // Note: final spins are in `spins`, not in `out_spins`"
    int *arr = new int[spins.size()];
    copy_array_from_device<int>(dev.n, dev.spins, arr);
    for (size_t i = 0; i < spins.size(); i++) {
        spins(i) = arr[i];
    }
    delete [] arr;
}

GPUSimple2DIsingModel::GPUSimple2DIsingModel(Simple2DIsingParams initial_params)
        : Simple2DIsingModel(initial_params) {

}

void GPUSimple2DIsingModel::execute_simulation_step_on_gpu(
        struct Simple2DModelDeviceMemoryParams &dev) const {
    // TODO: take in "AbstractDeviceMemoryParams"

    // Execute one step
    execute_one_step_simple(dev, 0);

    // Switch in spins (ptr) with out spins (ptr)
    int *tmp = dev.spins;
    dev.spins = dev.out_spins;
    dev.out_spins = tmp;

    // Execute one step
    execute_one_step_simple(dev, 0);

    // Switch in spins (ptr) with out spins (ptr)
    tmp = dev.spins;
    dev.spins = dev.out_spins;
    dev.out_spins = tmp;
}

struct Simple2DModelDeviceMemoryParams
GPUSimple2DIsingModel::prepare_gpu() const {
    struct Simple2DModelDeviceMemoryParams dev(2020ULL, spins.size());
    dev.spins = copy_vector_to_gpu<int, int>(spins);
    dev.xlen = params.xlen;
    dev.out_spins = copy_vector_to_gpu<int, int>(spins);
    dev.interaction = params.interaction;
    dev.external_field = params.external_field;
    dev.n = spins.size();
    dev.beta = 1. / (BOLTZMANN * params.temperature);
    dev.magnetic_moment = params.magnetic_moment;
    return dev;
}
