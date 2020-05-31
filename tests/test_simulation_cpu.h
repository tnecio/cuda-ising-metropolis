#ifndef ISING_TEST_SIMULATION_CPU_H
#define ISING_TEST_SIMULATION_CPU_H

#include "utils.h"
#include "test_simulation.h"

#include "../simulation_cpu.h"


TEST_SUITE_START(test_cpu_generalised_ising_model)
    // Use the same params as in test_simulation.h
    struct GeneralisedIsingParams params = prepare_params(10);
    CPUGeneralisedIsingModel model(params);
    TEST(test_run, model, params.external_field, 10);
    TEST(test_reset, model, params.initial_spins);
TEST_SUITE_END

#endif //ISING_TEST_SIMULATION_CPU_H
