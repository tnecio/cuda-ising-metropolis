#ifndef _ISING_SIMULATION_H_
#define _ISING_SIMULATION_H_

#include <vector>
#include <string>

const double BOLTZMANN = 1.0;

double calc_beta(double temperature);

enum IsingParamsErrors {
    ISING_PARAMS_OK = 0,
    INCONSISTENT_DIMENSIONS = 1,
    NONZERO_INTERACTION_OF_NONADJACENT_SPINS = 2,
    NEGATIVE_TEMPERATURE = 3
};

class IsingParams {
public:
    std::vector<int> initial_spins;
    double temperature = 0.0; //
    double magnetic_moment = 1.0; // mu

    explicit IsingParams(uint size) : initial_spins(size) {}

    virtual IsingParamsErrors check_validity() = 0;

    static std::string get_error_message(IsingParamsErrors errors);
};

class IsingModel {
protected:
    unsigned int step_count = 0;

    std::vector<int> spins; // current configuration

    double get_spin(size_t index) const;

    virtual double get_spin_energy(size_t index) const = 0;

public:
    const std::vector<int> &get_spins() const {
        return spins;
    }

    double get_total_energy() const;

    double get_mean_magnetisation() const;
    
    virtual double get_susceptibility() = 0;

    virtual void run(size_t max_steps) = 0;

    virtual void reset() = 0;
};

class Simple2DIsingParams : public IsingParams {
public:
    double interaction = 0.0; // J

    double external_field = 0.0; // B

    size_t xlen, ylen; // index = x + y * xlen

    IsingParamsErrors check_validity();

    Simple2DIsingParams(size_t xlen_, size_t ylen_)
            : IsingParams(xlen_ * ylen_), xlen(xlen_), ylen(ylen_) {}
};

class Simple2DIsingModel : public IsingModel {
    // Simple model from J. Tworzydlo's CMPP lecture summer 2019
    // -J \sum_{<i j>} s_i s_j - \mu B \sum_i s_i
    // where <i j> are direct neighbours in cardinal directions on a 2D grid

protected:
    const Simple2DIsingParams params;

    double get_spin_energy(size_t index) const;

public:
    explicit Simple2DIsingModel(Simple2DIsingParams params);

    double get_susceptibility();

    void reset();
};

#endif //_ISING_SIMULATION_H_