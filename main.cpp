#include "simulation_cpu.h"
#include "simulation_gpu.h"

void println(std::string line) {
    std::cout << line << std::endl;
}

void printerr(std::string line) {
    std::cerr << line << std::endl;
}

void print_usage() {
    printerr("Usage: ising [map steps xlen ylen J h T | plot steps xlen ylen J h]");
}

bool strings_equal(std::string a, std::string b) {
    return std::string(a).compare(b) == 0;
}

struct SimulationParams {
    size_t xlen, ylen, steps;
    double J, h;
    double T = 0.0;
};


void make_map(const struct SimulationParams &p) {
    Simple2DIsingParams params(p.xlen, p.ylen);
    params.external_field = p.h;
    params.interaction = p.J;
    params.initial_spins = std::vector<int>(p.xlen * p.ylen);
    PRNG prng(time(nullptr));
    for (size_t i = 0; i < p.xlen * p.ylen; i++) {
        params.initial_spins[i] = prng.random_double() < 0.5 ? 1 : -1;
    }
    GPUSimple2DIsingModel model(params);
    model.run(p.steps);
    std::vector<int> res = model.get_spins();

    std::cout << p.xlen << " " << p.ylen << " ";
    for (size_t i = 0; i < p.xlen * p.ylen; i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, const char **argv) {
    if (argc < 7 || argc > 8) {
        print_usage();
        return 0;
    }

    struct SimulationParams p;
    p.steps = std::stod(argv[2]);
    p.xlen = std::stod(argv[3]);
    p.ylen = std::stod(argv[4]);
    p.J = std::stod(argv[5]);
    p.h = std::stod(argv[6]);

    if (strings_equal(argv[1], "map")) {
        p.T = std::stod(argv[7]); // TODO: try, if not use default + docum.
        make_map(p);
    }
    else if (strings_equal(argv[1], "plot")) {

    }


    return 0;
}