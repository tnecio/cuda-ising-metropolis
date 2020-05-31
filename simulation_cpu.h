
#ifndef ISING_SIMULATION_CPU_H
#define ISING_SIMULATION_CPU_H

#include "simulation.h"

class PRNG {
private:
    std::random_device random_device;
    std::mt19937 random_engine;
    std::uniform_real_distribution<> real_dist;
    std::uniform_int_distribution<> int_dist;

public:
    explicit PRNG(uint int_limit);

    uint random_uint();

    double random_double();
};


class CPUGeneralisedIsingModel : public GeneralisedIsingModel {
private:
    PRNG prng;

    void flip_spin_stochastically();

    void flip_spin_deterministically(uint index, vector<bool> &new_spins);

    double get_spin_flip_prob(double energy) const;

public:
    CPUGeneralisedIsingModel(GeneralisedIsingParams initial_params);

    void run(uint no_steps);
};



#endif //ISING_SIMULATION_CPU_H
