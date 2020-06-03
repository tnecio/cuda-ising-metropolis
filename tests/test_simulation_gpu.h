#ifndef ISING_TEST_SIMULATION_GPU_H
#define ISING_TEST_SIMULATION_GPU_H

#include "utils.h"
#include "test_simulation.h"

#include "../simulation_gpu.h"


TEST_SUITE_START(test_gpu_generalised_ising_model)
    Simple2DIsingParams params2 = prepare_params_simple();
    GPUSimple2DIsingModel model2(params2);
    TEST(test_run_simple, model2);
TEST_SUITE_END

#endif //ISING_TEST_SIMULATION_GPU_H
