
#include <random>

#include "simulation_gpu.h"

#include "main_cpu.h"


void print_current_configuration(const int sqrt_size, const GeneralisedIsingModel &simulation) {
    const int size = sqrt_size * sqrt_size;
    const vector<int> &c = simulation.get_spins();
    for (int j = 0; j < size; j++) {
        std::cout << ((c[j] == 1) ? "↑" : ".");
        if (j % sqrt_size == sqrt_size - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void prepare_test_values(vector<int> &initial_configuration,
                         matrix<double> &J,
                         vector<double> &h,
                         size_t sqrt_size) {

    size_t size = sqrt_size * sqrt_size;

    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_real_distribution<> distribution(0., 1.);

    for (int i = 0; i < size; i++) {
        initial_configuration(i) = (0.5 < distribution(random_engine)) ? 1 : -1;
    }

    for (uint i = 0; i < J.size1(); i++) {
        for (int j = 0; j < J.size2(); j++) {
            J(i, j) = 0;
        }
    }
    // Sets interaction to be with closest neighbours
    for (uint i = 0; i < J.size1(); i++) {
        J(i, (i + 1) % J.size1()) = -1;
        J(i, (i - 1) % J.size1()) = -1;
        J(i, (i + sqrt_size) % J.size1()) = -1;
        J(i, (i - sqrt_size) % J.size1()) = -1;
    }

    const uint domain_count = 5; // No. of domains per magnetic orientation
    for (int i = 0; i < size; i++) {
        if ((i / sqrt_size) % (sqrt_size / domain_count) < (sqrt_size / (2 * domain_count))) {
            h(i) = -100;
        } else {
            h(i) = 100;
        }
    }
}

void main_cpu() {
    uint sqrt_size = 10;
    uint size = sqrt_size * sqrt_size;
    vector<int> initial_configuration(size);
    double T = 0.1;
    matrix<double> J(size, size);
    vector<double> h(size);
    prepare_test_values(initial_configuration, J, h, sqrt_size);
    GeneralisedIsingParams params;
    params.initial_spins = initial_configuration;
    params.external_field = h;
    params.magnetic_moment = 1.0;
    params.interaction = J;
    params.temperature = T;
    GPUGeneralisedIsingModel simulation(params);

    struct GeneralisedIsingParams params_copy = params;
    std::cout << params_copy.initial_spins(12) << std::endl;

    for (int i = 0; i < 20; i++) {
        print_current_configuration(sqrt_size, simulation);
        simulation.run(100);
    }
    print_current_configuration(sqrt_size, simulation);
}