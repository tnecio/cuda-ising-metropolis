#include <cmath>
#include <random>

#include "simulation_cpu.h"

PRNG::PRNG(uint int_limit) :
        random_engine(std::mt19937(random_device())),
        real_dist(std::uniform_real_distribution<>(0., 1.)),
        int_dist(std::uniform_int_distribution<>(0, int_limit - 1)) {
}

double PRNG::random_double() {
    return real_dist(random_engine);
}

uint PRNG::random_uint() {
    return int_dist(random_engine);
}

CPUGeneralisedIsingModel::CPUGeneralisedIsingModel(GeneralisedIsingParams initial_params) :
    GeneralisedIsingModel(initial_params),
    prng(initial_params.initial_spins.size()) {
}

void CPUGeneralisedIsingModel::run(uint no_steps) {
    // Sequential Metropolis algorithm
    for (uint i = 0; i < no_steps; i++) {
//        flip_spin_stochastically();
//        this->step_count += 1;
        vector<bool> new_spins;
        for (uint j = 0; j < spins.size(); j++) {
            flip_spin_deterministically(j, new_spins); // TMP
        }
        spins = new_spins;
    }
}

void CPUGeneralisedIsingModel::flip_spin_deterministically(uint index, vector<bool> &new_spins) {
    double energy = get_spin_energy(index);

    if (energy < 0  || prng.random_double() < get_spin_flip_prob(energy)) {
        new_spins(index) = !spins(index);
    }
}

void CPUGeneralisedIsingModel::flip_spin_stochastically() {
    uint index = prng.random_uint();
    double energy = get_spin_energy(index);

    if (energy < 0  || prng.random_double() < get_spin_flip_prob(energy)) {
        spins(index) = !spins(index);
    }
}

double CPUGeneralisedIsingModel::get_spin_flip_prob(double energy) const {
    double beta = 1. / (BOLTZMANN * params.temperature);
    return exp(-beta * energy);
}
