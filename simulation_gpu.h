#ifndef ISING_SIMULATION_GPU_H
#define ISING_SIMULATION_GPU_H

#include "simulation.h"

#include "simulation_gpu_cuda.cuh"


class GPUGeneralisedIsingModel : public GeneralisedIsingModel {
protected:
    struct DeviceMemoryParams prepare_gpu() const;

    void execute_simulation_step_on_gpu(
            struct DeviceMemoryParams dev) const;

    void read_spins_from_gpu(struct DeviceMemoryParams dev);

public:
    GPUGeneralisedIsingModel(GeneralisedIsingParams initial_params) :
    GeneralisedIsingModel(initial_params) { }

    void run(uint no_steps);
};

#endif //ISING_SIMULATION_GPU_H
