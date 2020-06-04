#ifndef ISING_SIMULATION_CPU_H
#define ISING_SIMULATION_CPU_H

#include <random>

#include "simulation.h"

class PRNG {
protected:
    std::mt19937 random_engine;
    std::uniform_real_distribution<> real_dist;

public:
    explicit PRNG(unsigned long long seed);

    double random_double();
};

class BufferedPRNG : public PRNG {
private:
    std::vector<double> buffer;
    size_t buffer_size;
    size_t usage_count = 0;

    void generate();
public:
    explicit BufferedPRNG(unsigned long long seed, size_t size);

    double random_double(size_t index);
};


class CPUSimple2DIsingModel : public Simple2DIsingModel {
private:
    PRNG prng;

    void flip_one_spin_stochastically(size_t index);

    void flip_spins_stochastically(int offset); // 0 or 1

    double get_spin_flip_prob(double energy) const;

public:
    CPUSimple2DIsingModel(Simple2DIsingParams initial_params);

    void run(size_t max_steps);
};



#endif //ISING_SIMULATION_CPU_H
