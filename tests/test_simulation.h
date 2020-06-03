#ifndef ISING_TEST_SIMULATION_H
#define ISING_TEST_SIMULATION_H

#include "utils.h"

#include "../simulation.h"


std::vector<int> prepare_initial_spins(uint sqrt_size) {
    uint size = sqrt_size * sqrt_size;
    std::vector<int> res(size);
    std::mt19937 random_engine(1230ULL);
    for (int i = 0; i < size; i++) {
        res[i] = 0.5 < random_double(random_engine) ? 1 : -1;
    }
    return res;
}

std::vector<double> prepare_external_field(uint sqrt_size) {
    std::vector<double> external_field(sqrt_size * sqrt_size);
    const uint no_domains = 5; // No. of domains per magnetic orientation
    for (int i = 0; i < sqrt_size * sqrt_size; i++) {
        if ((i / sqrt_size) % (sqrt_size / no_domains) <
            (sqrt_size / (2 * no_domains))) {
            external_field[i] = -100;
        } else {
            external_field[i] = 100;
        }
    }
    return external_field;
}

// Methods below to be used to test concrete implementations

void print_spins(const std::vector<int> &c) {
    uint sqrt_size = (uint) std::sqrt(c.size());
    for (int j = 0; j < c.size(); j++) {
        std::cout << ((c[j] == 1) ? "↑" : ".");
        if (j % sqrt_size == sqrt_size - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


Simple2DIsingParams prepare_params_simple() {
    // TODO: some better test :)
    Simple2DIsingParams params(10, 10);
    params.external_field = 7.0;
    params.interaction = 1.0;
    params.temperature = 0.0;

    std::vector<int> initial_spins(100);
    for (int i = 0; i < 100; i++) {
        initial_spins[i] = i / 10 > 7 ? 1 : -1;
    }
    params.initial_spins = initial_spins;
    return params;
}

int test_run_simple(Simple2DIsingModel &model) {
    print_spins(model.get_spins());
    model.run(1000);
    print_spins(model.get_spins());
    std::vector<int> spins = model.get_spins();
    for (int i = 0; i < 100; i++) {
        if (spins[i] != -1) {
            return fail("Simple2DModel run", "Bad spin");
        }
    }
    return success("Simple2DModel run");
}


#endif //ISING_TEST_SIMULATION_H
