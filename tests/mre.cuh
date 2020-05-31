#ifndef ISING_MRE_CUH
#define ISING_MRE_CUH

#include <curand.h>

class CudaPRNG {
    curandGenerator_t gen;
    int size = 0;

public:
    float *where;

    CudaPRNG(unsigned long long seed, int size_);

    curandStatus generate();
};

int test();

#endif //ISING_MRE_CUH
