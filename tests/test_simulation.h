#ifndef ISING_TEST_SIMULATION_H
#define ISING_TEST_SIMULATION_H

#include "utils.h"

#include "../simulation.h"

// GeneralisedIsingModel is abstract
class NotabstractGeneralisedIsingModel : public GeneralisedIsingModel {
public:
    void run(uint no_steps) override {
    }

    explicit NotabstractGeneralisedIsingModel(
            struct GeneralisedIsingParams params)
            : GeneralisedIsingModel(std::move(params)) {
    }
};

vector<int> prepare_initial_spins(uint sqrt_size) {
    uint size = sqrt_size * sqrt_size;
    vector<int> res(size);
    std::mt19937 random_engine(1230ULL);
    for (int i = 0; i < size; i++) {
        res(i) = 0.5 < random_double(random_engine) ? 1 : -1;
    }
    return res;
}

vector<double> prepare_external_field(uint sqrt_size) {
    vector<double> external_field(sqrt_size * sqrt_size);
    const uint no_domains = 5; // No. of domains per magnetic orientation
    for (int i = 0; i < sqrt_size * sqrt_size; i++) {
        if ((i / sqrt_size) % (sqrt_size / no_domains) <
            (sqrt_size / (2 * no_domains))) {
            external_field(i) = -100;
        } else {
            external_field(i) = 100;
        }
    }
    return external_field;
}

matrix<double> prepare_interaction(uint sqrt_size) {
    uint size = sqrt_size * sqrt_size;
    matrix<double> interaction(size, size);

    for (uint i = 0; i < interaction.size1(); i++) {
        for (int j = 0; j < interaction.size2(); j++) {
            interaction(i, j) = 0;
        }
    }

    // Sets interaction to be with closest neighbours
    for (uint i = 0; i < interaction.size1(); i++) {
        interaction(i, (i + 1) % interaction.size1()) = -1;
        interaction(i, (i - 1) % interaction.size1()) = -1;
        interaction(i, (i + sqrt_size) % interaction.size1()) = -1;
        interaction(i, (i - sqrt_size) % interaction.size1()) = -1;
    }

    return interaction;
}

struct GeneralisedIsingParams prepare_params(uint sqrt_size) {
    struct GeneralisedIsingParams params(sqrt_size * sqrt_size);
    params.magnetic_moment = 1.0;
    params.temperature = 0.1;
    params.initial_spins = prepare_initial_spins(sqrt_size);
    params.external_field = prepare_external_field(sqrt_size);
    params.interaction = prepare_interaction(sqrt_size);
    return params;
}

NotabstractGeneralisedIsingModel prepare_general_model(uint sqrt_size) {
    struct GeneralisedIsingParams params = prepare_params(sqrt_size);
    NotabstractGeneralisedIsingModel res(params);
    return res;
}

int test_get_total_energy(const NotabstractGeneralisedIsingModel &model) {
    double actual = model.get_total_energy();
    double expected = 400.0;
    double abs_diff = std::abs(expected - actual);
    std::ostringstream msg;
    msg << "Expected:" << expected << " Actual:" << actual;
    return ASSERT("get_total_energy", abs_diff < 0.000001, msg.str());
}

int test_get_spins(const NotabstractGeneralisedIsingModel &model) {
    uint size = model.get_spins().size();
    vector<int> res(size);
    std::mt19937 random_engine(1230ULL);
    for (int i = 0; i < size; i++) {
        res(i) = 0.5 < random_double(random_engine) ? 1 : -1;
        if (res(i) != model.get_spins()(i)) {
            return fail("get_spins", "Not equal spins");
        }
    }
    return success("get_spins");
}

TEST_SUITE_START(test_generalised_ising_model)
    NotabstractGeneralisedIsingModel model = prepare_general_model(10);
    TEST(test_get_total_energy, model);
    TEST(test_get_spins, model);
TEST_SUITE_END

// Methods below to be used to test concrete implementations

void print_spins(const vector<int> &c) {
    uint sqrt_size = (uint) std::sqrt(c.size());
    for (int j = 0; j < c.size(); j++) {
        std::cout << ((c[j] == 1) ? "↑" : ".");
        if (j % sqrt_size == sqrt_size - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

int test_run(GeneralisedIsingModel &model,
             const vector<double> &external_field) {
    model.run(1000);

    vector<int> res = model.get_spins();
    vector<int> expected(res.size());
    for (size_t i = 0; i < external_field.size(); i++) {
        expected(i) = (external_field(i) > 0) ? 1 : -1;
    }

    for (size_t i = 0; i < expected.size(); i++) {
        if (expected(i) == res(i)) {
            print_spins(model.get_spins());
            print_spins(expected);
            return fail("GeneralisedModel run",
                        "Spin opposite to magnetic field");
        }
    }
    return success("GeneralisedModel run");
}

Simple2DIsingParams prepare_params_simple() {
    Simple2DIsingParams params(10, 10);
    params.external_field = 7.0;
    params.interaction = 1.0;
    params.temperature = 0.0;

    vector<int> initial_spins(100);
    for (int i = 0; i < 100; i++) {
        initial_spins(i) = i / 10 > 7 ? 1 : -1;
    }
    params.initial_spins = initial_spins;
    return params;
}

int test_run_simple(Simple2DIsingModel &model) {
    print_spins(model.get_spins());
    model.run(100);
    print_spins(model.get_spins());
    vector<int> spins = model.get_spins();
    for (int i = 0; i < 100; i++) {
        if (spins(i) != -1) {
            return fail("Simple2DModel run", "Bad spin");
        }
    }
    return success("Simple2DModel run");
}

int test_reset(GeneralisedIsingModel &model,
               const vector<int> &initial_spins) {
    model.reset();
    model.run(1000);
    vector<int> res = model.get_spins();
    model.reset();
    vector<int> reset = model.get_spins();
    for (size_t i = 0; i < res.size(); i++) {
        if (reset(i) != initial_spins(i)) {
            return fail("GeneralisedModel reset",
                        "Spin different to initial after reset");
        }
    }
    return success("GeneralisedModel reset");
}

#endif //ISING_TEST_SIMULATION_H
