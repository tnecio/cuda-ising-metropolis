
#include "simulation_gpu.h"

void GPUGeneralisedIsingModel::run(uint no_steps) {
    DeviceMemoryParams dev = prepare_gpu();
    // TODO: separate preparation from running for speed-measurement purposes
    for (uint i = 0; i < no_steps; i++) {
        execute_simulation_step_on_gpu(dev);
        this->step_count += 1;
    }
    read_spins_from_gpu(dev);
}

void GPUGeneralisedIsingModel::read_spins_from_gpu(
        struct DeviceMemoryParams dev) {
    // Note: final spins are in `spins`, not in `out_spins`"
    int *arr = new int[spins.size()];
    copy_array_from_device<int>(dev.n, dev.spins, arr);
    for (size_t i = 0; i < spins.size(); i++) {
        spins(i) = (bool) arr[i];
    }
}

void GPUGeneralisedIsingModel::execute_simulation_step_on_gpu(
        struct DeviceMemoryParams dev) const {
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

int *copy_bool_vector_to_gpu(const vector<bool> &v) {
    int *host_arr = vector_to_array<bool, int>(v);
    int *dev_arr = copy_array_to_device<int>(v.size(), host_arr);
    delete[] host_arr;
    return dev_arr;
}

float *copy_float_vector_to_gpu(const vector<double> &v) {
    float *host_arr = vector_to_array<double, float>(v);
    float *dev_arr = copy_array_to_device<float>(v.size(), host_arr);
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

struct DeviceMemoryParams GPUGeneralisedIsingModel::prepare_gpu() const {
    struct DeviceMemoryParams dev(2020ULL, spins.size());
    dev.spins = copy_bool_vector_to_gpu(spins);
    dev.out_spins = copy_bool_vector_to_gpu(spins);
    dev.interaction = copy_matrix_to_gpu(params.interaction);
    dev.external_field = copy_float_vector_to_gpu(params.external_field);
    dev.n = spins.size();
    dev.beta = 1. / (BOLTZMANN * params.temperature);
    dev.magnetic_moment = params.magnetic_moment;
    return dev;
}