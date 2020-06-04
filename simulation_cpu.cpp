#include <cmath>
#include <random>

#include "simulation_cpu.h"

PRNG::PRNG(unsigned long long seed)
        : random_engine(std::mt19937(seed)),
          real_dist(std::uniform_real_distribution<>(0., 1.)) {
}

double PRNG::random_double() {
    return real_dist(random_engine);
}

BufferedPRNG::BufferedPRNG(unsigned long long int seed, size_t size)
        : PRNG(seed), buffer(size), buffer_size(size) {
    generate();
}

double BufferedPRNG::random_double(size_t index) {
    double res = buffer[index];
    usage_count++;
    if (usage_count == buffer_size) {
        usage_count = 0;
        generate();
    }
    return res;
}

void BufferedPRNG::generate() {
    for (size_t i = 0; i < buffer_size; i++) {
        buffer[i] = real_dist(random_engine);
    }
}

CPUSimple2DIsingModel::CPUSimple2DIsingModel(
        Simple2DIsingParams initial_params)
        : Simple2DIsingModel(initial_params),
          prng(2020ULL) {
}

void CPUSimple2DIsingModel::run(size_t max_steps) {
    for (size_t i = 0; i < max_steps; i++) {
        flip_spins_stochastically(0);
        flip_spins_stochastically(1);
        step_count++;
    }
}

void CPUSimple2DIsingModel::flip_spins_stochastically(int offset) {
    // offset = 0 or 1 (think BLACK/WHITE field on a CHESSBOARD)
    for (size_t i = 0; i < spins.size(); i++) {
        size_t x = i % params.xlen;
        size_t y = i / params.xlen;
        bool is_odd_position = (x + y + offset) % 2;
        if (is_odd_position) {
            flip_one_spin_stochastically(i);
        }
    }

}

void CPUSimple2DIsingModel::flip_one_spin_stochastically(size_t index) {
    // \Delta E = E_after_flip - E_now = -spin_energy_now - spin_energy_now
    double delta_energy = -2 * get_spin_energy(index);
    if (delta_energy < 0 ||
        prng.random_double() < get_spin_flip_prob(delta_energy)) {
        spins[index] = -spins[index];
    }
}

double CPUSimple2DIsingModel::get_spin_flip_prob(double energy) const {
    double beta = 1. / (BOLTZMANN * params.temperature);
    return exp(-beta * energy);
}
