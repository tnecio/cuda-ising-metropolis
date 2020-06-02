#include "simulation_gpu_cuda.cuh"

template<typename T>
T *copy_array_to_device(int size, T *from) {
    T *to;
    cudaMalloc(&to, size * sizeof(T));
    cudaMemcpy(to, from, size * sizeof(T), cudaMemcpyHostToDevice);
    return to;
}

template float *copy_array_to_device<float>(int, float *);

template int *copy_array_to_device<int>(int, int *);

template<typename T>
cudaError copy_array_from_device(int size, T *from, T *to) {
    return cudaMemcpy(to, from, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template cudaError copy_array_from_device<int>(int, int *, int *);

template cudaError copy_array_from_device<float>(int, float *, float *);

CudaPRNG::CudaPRNG(unsigned long long int seed, int size_)
        : size(size_) {
    curandStatus cs = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandStatus cs2 = curandSetPseudoRandomGeneratorSeed(gen, seed);
    cudaError ce = cudaMalloc((void **) &where, size_ * sizeof(float));
}

curandStatus CudaPRNG::generate() {
    return curandGenerateUniform(gen, where, size);
}

// These function are direct analogues of those in simulation / simulation_cpu

__device__
float get_spin(int i, int *in_spins) {
    return in_spins[i] ? 1.0 : -1.0;
}

__device__
float get_spin_energy(int i, struct GeneralisedModelDeviceMemoryParams dev) {
    float res = 0;
    float spin = get_spin(i, dev.spins);
    res += -dev.magnetic_moment *
           dev.external_field[i] * spin;
    for (int j = 0; j < dev.n; j++) {
        res += dev.interaction[i * dev.n + j] * spin * get_spin(j, dev.spins);
    }
    return res;
}

__device__
float get_spin_energy_simple(int i, struct Simple2DModelDeviceMemoryParams dev) {
    float res = 0;
    float spin = get_spin(i, dev.spins);
    res += -dev.magnetic_moment *
           dev.external_field * spin;
    for (int j = 0; j < dev.n; j++) {
        res += dev.interaction * spin * get_spin(j, dev.spins);
    }
    return res;
}

__device__
float get_spin_flip_prob(float energy, float beta) {
    return exp(-beta * energy);
}

__global__
void flip_spins_stochastically(struct GeneralisedModelDeviceMemoryParams dev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dev.n) {
        float energy = get_spin_energy(i, dev);
        if (energy < 0 ||
            dev.prng.where[i] < get_spin_flip_prob(energy, dev.beta)) {
            dev.out_spins[i] = -dev.spins[i];
        } else {
            dev.out_spins[i] = dev.spins[i];
        }
    }
}

__global__
void flip_spins_stochastically_simple(struct Simple2DModelDeviceMemoryParams dev, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int is_odd = (i + (i / dev.xlen * !(dev.xlen % 2)) + offset) % 2;
    if (i < dev.n && is_odd) {
        float energy = get_spin_energy_simple(i, dev);
        if (energy < 0 ||
            dev.prng.where[i] < get_spin_flip_prob(energy, dev.beta)) {
            dev.out_spins[i] = -dev.spins[i];
        } else {
            dev.out_spins[i] = dev.spins[i];
        }
    }
}


void execute_one_step(struct GeneralisedModelDeviceMemoryParams &dev) {
    int blocks = (dev.n + 255) / 256;
    int THREADS_PER_BLOCK = 256;
    dev.prng.generate();
    flip_spins_stochastically<<<blocks, THREADS_PER_BLOCK>>>(dev);
}

void execute_one_step_simple(Simple2DModelDeviceMemoryParams &dev, int offset) {
    int blocks = (dev.n + 255) / 256;
    int THREADS_PER_BLOCK = 256;
    dev.prng.generate();
    flip_spins_stochastically_simple<<<blocks, THREADS_PER_BLOCK>>>(dev, offset);
}
