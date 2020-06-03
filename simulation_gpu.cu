#include "simulation_gpu.h"

void cuda_failure(std::string stmt, int ce) {
    std::cerr << "CUDA Failure: [" << ce << "] in line: " << stmt << std::endl;
}

CudaPRNG::CudaPRNG(unsigned long long int seed, int size_)
        : size(size_) {
    CUDA_HANDLERR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CUDA_HANDLERR(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CUDA_HANDLERR(cudaMalloc((void **) &where, size_ * sizeof(float)));
}

void CudaPRNG::generate() {
    CUDA_HANDLERR(curandGenerateUniform(gen, where, size));
}

Simple2DIsingParamsDevice::Simple2DIsingParamsDevice(const Simple2DIsingParams &params) {
    // Copy constructor from CPU version of params
    spins = CudaDevice::copy_array_to_device<int>(
            params.initial_spins.size(),
            &(params.initial_spins[0])
    );
    n = (int) params.initial_spins.size();
    beta = (float) calc_beta(params.temperature);
    magnetic_moment = (float) params.magnetic_moment;

    xlen = (int) params.xlen;
    ylen = (int) params.ylen;

    interaction = (float) params.interaction;
    external_field = (float) params.external_field;
}

// These kernels are direct analogues of methods in the CPU simulation
__device__
float get_spin_energy_simple(int i,
                             const struct Simple2DIsingParamsDevice &dev) {
    // TODO: generalise over dimensions

    float res = 0;
    float spin = dev.spins[i];
    res += -dev.magnetic_moment *
           dev.external_field * spin;

    int xlen = dev.xlen;
    int ylen = dev.ylen;
    int x = i % xlen;
    int y = i / xlen;

    int left = (x - 1) % xlen + y * xlen;
    int right = (x + 1) % xlen + y * xlen;
    int top = x + (y - 1) % ylen * xlen;
    int down = x + (y + 1) % ylen * xlen;

    res += -dev.interaction * spin * dev.spins[left];
    res += -dev.interaction * spin * dev.spins[right];
    res += -dev.interaction * spin * dev.spins[top];
    res += -dev.interaction * spin * dev.spins[down];

    return res;
}

__device__
float get_spin_flip_prob(float energy, float beta) {
    return exp(-beta * energy);
}

__device__
void
flip_one_spin_stochastically(struct Simple2DIsingParamsDevice &dev,
                             size_t i,
                             float *random_floats) {
    float energy = get_spin_energy_simple(i, dev);
    if (energy < 0 ||
        random_floats[i] < get_spin_flip_prob(energy, dev.beta)) {
        dev.spins[i] = -dev.spins[i];
    }
}

__global__
void
flip_spins_stochastically(struct Simple2DIsingParamsDevice dev,
                          int offset,
                          float *random_floats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int is_odd_position = (i + (i / dev.xlen * !(dev.xlen % 2)) + offset) % 2;
    if (i < dev.n && is_odd_position) {
        flip_one_spin_stochastically(dev, i, random_floats);
    }
}


void GPUSimple2DIsingModel::run(size_t max_steps) {
    for (size_t i = 0; i < max_steps; i++) {
        execute_simulation_step_on_gpu();
        step_count++;
    }
    read_spins_from_gpu();
}

void GPUSimple2DIsingModel::execute_simulation_step_on_gpu() {
    prng.generate();
    int blocks = (dev.n + 255) / 256;
    int THREADS_PER_BLOCK = 256;
    flip_spins_stochastically<<<blocks, THREADS_PER_BLOCK>>>(dev, 0,
                                                             prng.where);
    flip_spins_stochastically<<<blocks, THREADS_PER_BLOCK>>>(dev, 1,
                                                             prng.where);
}


void GPUSimple2DIsingModel::read_spins_from_gpu() {
    CudaDevice::copy_array_from_device<int>(dev.n, dev.spins, &(spins[0]));
}


template<typename DevType, typename HostType>
DevType *CudaDevice::copy_array_to_device_with_conversion(
        int size, const HostType *from) {
    DevType *conv = (DevType *) malloc(sizeof(DevType) * size); // TODO errhandle
    for (size_t i = 0; i < size; i++) {
        conv[i] = (DevType) from[i];
    }
    CudaDevice::copy_array_to_device(size, conv);
    free(conv);
}

template<typename DevType>
DevType *CudaDevice::copy_array_to_device(int size, const DevType *from) {
    DevType *to;
    CUDA_HANDLERR(cudaMalloc(&to, size * sizeof(DevType)));
    CUDA_HANDLERR(cudaMemcpy(to, from, size * sizeof(DevType), cudaMemcpyHostToDevice));
    return to;
}

template float *
CudaDevice::copy_array_to_device<float>(int size, const float *from);

template<typename DevType, typename HostType>
void CudaDevice::copy_array_from_device_with_conversion(
        int size, const DevType *from, HostType *to) {
    DevType *conv = (DevType *) malloc(sizeof(DevType) * size);
    CudaDevice::copy_array_from_device(size, from, conv);

    if (conv != nullptr) {
        for (size_t i = 0; i < size; i++) {
            to[i] = (HostType) conv[i];
        }
    }

    free(conv);
}

template<typename T>
void CudaDevice::copy_array_from_device(int size, const T *from, T *to) {
    CUDA_HANDLERR(cudaMemcpy(to, from, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template void
CudaDevice::copy_array_from_device<float>(int, const float *, float *);

void CudaDevice::free_array_from_device(void *arr) {
    CUDA_HANDLERR(cudaFree(arr));
}
