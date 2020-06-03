#ifndef ISING_SIMULATION_GPU_H
#define ISING_SIMULATION_GPU_H

#include "simulation.h"

#include "simulation_gpu_cuda.cuh"

//
//class GPUGeneralisedIsingModel : public GeneralisedIsingModel {
//protected:
//    struct GeneralisedModelDeviceMemoryParams prepare_gpu() const;
//
//    void execute_simulation_step_on_gpu(
//            struct GeneralisedModelDeviceMemoryParams &dev) const;
//
//    void read_spins_from_gpu(struct GeneralisedModelDeviceMemoryParams dev);
//
//public:
//    GPUGeneralisedIsingModel(GeneralisedIsingParams initial_params) :
//    GeneralisedIsingModel(initial_params) { }
//
//    void run(uint no_steps);
//};

class GPUSimple2DIsingModel : public Simple2DIsingModel {
protected:
    struct Simple2DModelDeviceMemoryParams prepare_gpu() const;

    // TODO: Make class "GPUModel"; double inheritance

    void execute_simulation_step_on_gpu(
            struct Simple2DModelDeviceMemoryParams &dev) const;

    void read_spins_from_gpu(struct Simple2DModelDeviceMemoryParams dev);

public:
    GPUSimple2DIsingModel(Simple2DIsingParams initial_params);

    void run(uint no_steps);
};

#endif //ISING_SIMULATION_GPU_H
