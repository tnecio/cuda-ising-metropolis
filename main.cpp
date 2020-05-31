#include "main_cpu.h"
#include "main_gpu.cuh"

int main() {
    main_cpu();
    main_gpu();
    return 0;
}