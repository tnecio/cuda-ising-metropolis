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

CPUGeneralisedIsingModel::CPUGeneralisedIsingModel(
        GeneralisedIsingParams initial_params) :
        GeneralisedIsingModel(initial_params),
        prng(initial_params.initial_spins.size()) {
}

void CPUGeneralisedIsingModel::run(uint no_steps) {
    // Sequential Metropolis algorithm
    for (uint i = 0; i < no_steps; i++) {
//        flip_spin_stochastically();
//        this->step_count += 1;
        vector<int> new_spins(spins.size());
        for (uint j = 0; j < spins.size(); j++) {
//            flip_spin_deterministically(j, new_spins); // TMP
            flip_spin_stochastically();
        }
        spins = new_spins;
    }
}

void CPUGeneralisedIsingModel::flip_spin_deterministically(uint index,
                                                           vector<int> &new_spins) {
    double energy = get_spin_energy(index);

    if (energy < 0 || prng.random_double() < get_spin_flip_prob(energy)) {
        new_spins(index) = -spins(index);
    }
}

void CPUGeneralisedIsingModel::flip_spin_stochastically() {
    uint index = prng.random_uint();
    double energy = get_spin_energy(index);

    if (energy < 0 || prng.random_double() < get_spin_flip_prob(energy)) {
        spins(index) = -spins(index);
    }
}

double CPUGeneralisedIsingModel::get_spin_flip_prob(double energy) const {
    double beta = 1. / (BOLTZMANN * params.temperature);
    return exp(-beta * energy);
}

CPUSimple2DIsingModel::CPUSimple2DIsingModel(
        Simple2DIsingParams initial_params)
        : Simple2DIsingModel(initial_params),
          prng(initial_params.initial_spins.size()) {
}

void CPUSimple2DIsingModel::run(uint no_steps) {
    for (uint i = 0; i < no_steps; i++) {
        flip_spins_stochastically(0);
        flip_spins_stochastically(1);
        step_count++;
    }
}

void CPUSimple2DIsingModel::flip_spins_stochastically(int offset) {
    // offset = 0 or 1 (think BLACK/WHITE field on a CHESSBOARD)
    for (size_t i = 0; i < spins.size(); i++) {
        bool is_odd_position =
                (i + i / params.xlen * !(params.xlen % 2) + offset) % 2;
        if (is_odd_position) {
            flip_one_spin_stochastically(i);
        }
    }
}

void CPUSimple2DIsingModel::flip_one_spin_stochastically(size_t index) {
    double energy = get_spin_energy(index);
    if (energy < 0 || prng.random_double() < get_spin_flip_prob(energy)) {
        spins(index) = -spins(index);
    }
}

double CPUSimple2DIsingModel::get_spin_flip_prob(double energy) const {
    // todo: fix this copy-paste from CPUGeneralisedIsingModel
    double beta = 1. / (BOLTZMANN * params.temperature);
    return exp(-beta * energy);
}
