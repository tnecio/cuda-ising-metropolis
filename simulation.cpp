#include <stdexcept>

#include "simulation.h"


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
    return spins(index);
}


IsingParamsErrors GeneralisedIsingParams::check_validity() {
    if (!all_sizes_equal()) {
        return INCONSISTENT_DIMENSIONS;
    } else if (temperature < 0) {
        return NEGATIVE_TEMPERATURE;
    } else {
        return ISING_PARAMS_OK;
    }
}

bool GeneralisedIsingParams::all_sizes_equal() {
    return (initial_spins.size() == external_field.size()
            && initial_spins.size() == interaction.size1()
            && initial_spins.size() == interaction.size2());
}


GeneralisedIsingModel::GeneralisedIsingModel(
        GeneralisedIsingParams initial_params)
        : params(initial_params) {
    IsingParamsErrors validation = initial_params.check_validity();
    if (validation == ISING_PARAMS_OK) {
        this->spins = initial_params.initial_spins;
    } else {
        throw std::runtime_error(IsingParams::get_error_message(validation));
    }
}

void GeneralisedIsingModel::reset() {
    spins = params.initial_spins;
    step_count = 0;
}

inline double GeneralisedIsingModel::get_spin_energy(size_t index) const {
    double res = 0;
    double spin = get_spin(index);
    res += -params.magnetic_moment *
           params.external_field(index) * spin;
    for (uint j = 0; j < spins.size(); j++) {
        res += -params.interaction(index, j) * spin * get_spin(j);
    }
    return res;
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
