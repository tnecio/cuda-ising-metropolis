#include "test_simulation_cpu.h"
#include "test_simulation_gpu.h"
#include "test_simulation_gpu_cuda.h"


int main() {
    int res = 0;
    res += test_cpu_generalised_ising_model();
    res += test_gpu_generalised_ising_model();
    res += test_gpu_cuda_functions();
    return res;
}