#include "simulation_cpu.h"

#ifdef GPU_ISING
#include "simulation_gpu.h"
#define MODEL GPUSimple2DIsingModel
#else
#define MODEL CPUSimple2DIsingModel
#endif

#include <chrono>
#include <iostream>

void println(std::string line) {
    std::cout << line << std::endl;
}

void printerr(std::string line) {
    std::cerr << line << std::endl;
}

void print_usage() {
    printerr(
            "Usage: ising [map | plot | time] steps xlen ylen J h [T (if map/time) | Tmax (if plot)] seed");
}

bool strings_equal(std::string a, std::string b) {
    return std::string(a).compare(b) == 0;
}

struct SimulationParams {
    size_t xlen, ylen, steps;
    double J, h;
    double T = 0.0;
    unsigned long long seed = 2020ULL;
};


MODEL prepare_model(const struct SimulationParams &p) {
    Simple2DIsingParams params(p.xlen, p.ylen);
    params.external_field = p.h;
    params.interaction = p.J;
    params.initial_spins = std::vector<int>(p.xlen * p.ylen);
    PRNG prng(p.seed);
    for (size_t i = 0; i < p.xlen * p.ylen; i++) {
        params.initial_spins[i] = prng.random_double() < 0.5 ? 1 : -1;
    }
    params.temperature = p.T;
    return MODEL(params);
}

void make_map(const struct SimulationParams &p) {
    MODEL model = prepare_model(p);
    model.run(p.steps);
    std::vector<int> res = model.get_spins();

    std::cout << p.xlen << " " << p.ylen << " ";
    for (size_t i = 0; i < p.xlen * p.ylen; i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;
}

void make_time(const struct SimulationParams &p) {
    MODEL model = prepare_model(p);

    unsigned long time_us = 0;
    for (int i = 0; i < 10; i++) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        model.run(p.steps);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                end - begin).count();
    }

    std::cout << time_us / 10 << " [µs]" << std::endl;
    std::cout << time_us / 10000 << " [ms]" << std::endl;
    std::cout << time_us / 10000000 << " [s]" << std::endl;
}


void make_plot(struct SimulationParams &p) {
    std::cout << "Temperature,Energy,Magnetisation,Susceptibility,SpecificHeat" << std::endl;
    double T_max = p.T;
    for (double T = 0; T < T_max; T += 0.1) {
        p.T = T;
        MODEL model = prepare_model(p);
        model.run(p.steps);
        std::cout << T << "," << model.get_total_energy() << ","
                  << model.get_mean_magnetisation() << ","
                  << model.get_susceptibility() << ","
                  << model.get_specific_heat() << std::endl;
    }
}

int main(int argc, const char **argv) {
    if (argc != 9) {
        print_usage();
        return 0;
    }

    struct SimulationParams p;
    p.steps = std::stod(argv[2]);
    p.xlen = std::stod(argv[3]);
    p.ylen = std::stod(argv[4]);
    p.J = std::stod(argv[5]);
    p.h = std::stod(argv[6]);
    p.T = std::stod(argv[7]);
    p.seed = std::stoull(argv[8]);

    if (strings_equal(argv[1], "map")) {
        make_map(p);
    } else if (strings_equal(argv[1], "plot")) {
        make_plot(p);
    } else if (strings_equal(argv[1], "time")) {
        make_time(p);
    }

    return 0;
}
