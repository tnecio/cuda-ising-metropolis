#include <stdexcept>

#include "simulation.h"

GeneralisedIsingModel::GeneralisedIsingModel(
        struct GeneralisedIsingParams initial_params) {
    IsingModelParamsErrors validation = are_params_valid(initial_params);
    if (validation == OK) {
        this->params = initial_params;
        this->spins = initial_params.initial_spins;
    } else {
        throw std::runtime_error(get_error_message(validation));
    }
}

IsingModelParamsErrors
GeneralisedIsingModel::are_params_valid(GeneralisedIsingParams &params) {
    if (!all_sizes_equal(params)) {
        return INCONSISTENT_DIMENSIONS;
    } else if (params.temperature < 0) {
        return NEGATIVE_TEMPERATURE;
    } else {
        return OK;
    }
}

bool GeneralisedIsingModel::all_sizes_equal(GeneralisedIsingParams &params) {
    return (params.initial_spins.size() == params.external_field.size()
            && params.initial_spins.size() == params.interaction.size1()
            && params.initial_spins.size() == params.interaction.size2());
}

std::string GeneralisedIsingModel::get_error_message(IsingModelParamsErrors e) {
    switch (e) {
        case OK:
            return "OK";
        case INCONSISTENT_DIMENSIONS:
            return "Inconsistent dimensions of params";
        case BAD_XSIZE:
            return "`Spins` size not divisible by x-size";
        case NONZERO_INTERACTION_OF_NONADJACENT_SPINS:
            return "Non-zero interaction of non-adjacent spins in the 2D grid";
        case NEGATIVE_TEMPERATURE:
            return "Negative temperature";
    }
}

double GeneralisedIsingModel::get_total_energy() const {
    double res = 0;
    for (size_t i = 0; i < spins.size(); i++) {
        res += get_spin_energy(i);
    }
    return res;
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

inline double GeneralisedIsingModel::get_spin(size_t index) const {
    return spins(index) ? 1. : -1.;
}

TwoDimensionalIsingModel::TwoDimensionalIsingModel(
        struct TwoDimensionalIsingParams params) :
        GeneralisedIsingModel(params) {
}
