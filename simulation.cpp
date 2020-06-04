#include <stdexcept>

#include "simulation.h"

double calc_beta(double temperature) {
    return 1.0 / (BOLTZMANN * temperature);
}

std::string IsingParams::get_error_message(IsingParamsErrors e) {
    switch (e) {
        case ISING_PARAMS_OK:
            return "OK";
        case INCONSISTENT_DIMENSIONS:
            return "Inconsistent dimensions of params";
        case NONZERO_INTERACTION_OF_NONADJACENT_SPINS:
            return "Non-zero interaction of non-adjacent spins in the grid";
        case NEGATIVE_TEMPERATURE:
            return "Negative temperature";
    }
}

double IsingModel::get_total_energy() const {
    double res = 0;
    for (size_t i = 0; i < spins.size(); i++) {
        res += get_spin_energy(i);
    }
    return res;
}


inline double IsingModel::get_spin(size_t index) const {
    return spins[index];
}

double IsingModel::get_mean_magnetisation() const {
    double magnetisation_sum = 0;
    for (size_t i = 0; i < spins.size(); i++) {
        magnetisation_sum += spins[i];
    }
    return magnetisation_sum / spins.size();
}

Simple2DIsingModel::Simple2DIsingModel(Simple2DIsingParams initial_params)
: params(initial_params) {
    // Must be copy-pasted basically since you can't put this in base
    // abstract class in C++ (params can't be abstract :<)
    IsingParamsErrors validation = initial_params.check_validity();
    if (validation == ISING_PARAMS_OK) {
        this->spins = initial_params.initial_spins;
    } else {
        throw std::runtime_error(IsingParams::get_error_message(validation));
    }
}

IsingParamsErrors Simple2DIsingParams::check_validity() {
    if (temperature < 0) {
        return NEGATIVE_TEMPERATURE;
    } else if (xlen * ylen != initial_spins.size()) {
        return INCONSISTENT_DIMENSIONS;
    }
    return ISING_PARAMS_OK;
}

double Simple2DIsingModel::get_spin_energy(size_t index) const {
    double res = 0;
    double spin = get_spin(index);
    res += -params.magnetic_moment *
           params.external_field * spin;

    // calculate neighbours
    size_t xlen = params.xlen;
    size_t ylen = params.ylen;
    size_t x = index % xlen;
    size_t y = index / xlen;

    size_t left = (x - 1) % xlen + y * xlen;
    size_t right = (x + 1) % xlen + y * xlen;
    size_t top = x + (y - 1) % ylen * xlen;
    size_t down = x + (y + 1) % ylen * xlen;

    res += -params.interaction * spin * get_spin(left);
    res += -params.interaction * spin * get_spin(right);
    res += -params.interaction * spin * get_spin(top);
    res += -params.interaction * spin * get_spin(down);

    return res;
}

void Simple2DIsingModel::reset() {
    spins = params.initial_spins;
    step_count = 0;
}

double Simple2DIsingModel::get_susceptibility() {
    double expected_square_magn = 1;  // (1**2 = (-1)**2 = 1)
    double expected_magn = get_mean_magnetisation();
    double expected_magn_squared = expected_magn * expected_magn;
    double temperature_sq = params.temperature * params.temperature;

    return (expected_square_magn - expected_magn_squared) / (temperature_sq);
}

double Simple2DIsingModel::get_specific_heat() {
    double sum_energy = get_total_energy();
    double sum_energy_sq = 0;
    for (size_t i = 0; i < spins.size(); i++) {
        double spin_energy = get_spin_energy(i);
        sum_energy_sq += spin_energy * spin_energy;
    }

    double n = spins.size();
    double expected_square_energy = sum_energy_sq / n;
    double expected_energy_squared = (sum_energy / n) * (sum_energy / n);
    double temperature_sq = params.temperature * params.temperature;

    return (expected_square_energy - expected_energy_squared) / (temperature_sq);
}