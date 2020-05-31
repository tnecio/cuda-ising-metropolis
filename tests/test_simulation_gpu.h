#ifndef ISING_TEST_SIMULATION_GPU_H
#define ISING_TEST_SIMULATION_GPU_H

#include "utils.h"
#include "test_simulation.h"

#include "../simulation_gpu.h"


TEST_SUITE_START(test_gpu_generalised_ising_model)
    // Use the same params as in test_simulation.h
    struct GeneralisedIsingParams params = prepare_params(10);
    GPUGeneralisedIsingModel model(params);
    TEST(test_run, model, params.external_field, 10);
    TEST(test_reset, model, params.initial_spins);
TEST_SUITE_END

#endif //ISING_TEST_SIMULATION_GPU_H
