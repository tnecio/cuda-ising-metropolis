#include <iostream>

#include "mre.cuh"

template<typename T>
cudaError copy_array_from_device(int size, T *from, T *to) {
    return cudaMemcpy(to, from, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template cudaError copy_array_from_device<int>(int, int *, int *);
template cudaError copy_array_from_device<float>(int, float *, float *);

CudaPRNG::CudaPRNG(unsigned long long seed, int size_) {
     curandStatus cs = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandStatus cs2 = curandSetPseudoRandomGeneratorSeed(gen, seed);
    cudaError ce = cudaMalloc((void **) &where, size_ * sizeof(float));

}

int test() {
    CudaPRNG cuda_prng(2020ULL, 1024);
    curandStatus cs = cuda_prng.generate();

    float arr[1024];
    copy_array_from_device(1024, cuda_prng.where, arr);

    for (int i = 0; i < 10; i++) {
        std::cout << i << ": " << arr[i] << std::endl;
    }

    return 0;
}

curandStatus CudaPRNG::generate() {
    return curandGenerateUniform(gen, where, size);
}
