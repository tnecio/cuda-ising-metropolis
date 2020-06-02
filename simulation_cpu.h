
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

    void flip_spin_deterministically(uint index, vector<int> &new_spins);

    double get_spin_flip_prob(double energy) const;

public:
    CPUGeneralisedIsingModel(GeneralisedIsingParams initial_params);

    void run(uint no_steps);
};

class CPUSimple2DIsingModel : public Simple2DIsingModel {
private:
    PRNG prng;

    void flip_one_spin_stochastically(size_t index);

    void flip_spins_stochastically(int offset); // 0 or 1

    double get_spin_flip_prob(double energy) const;

public:
    CPUSimple2DIsingModel(Simple2DIsingParams initial_params);

    void run(uint no_steps);
};



#endif //ISING_SIMULATION_CPU_H
