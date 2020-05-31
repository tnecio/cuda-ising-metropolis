#ifndef _ISING_SIMULATION_H_
#define _ISING_SIMULATION_H_

#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;

const double BOLTZMANN = 1.0;

struct GeneralisedIsingParams {
    vector<bool> initial_spins;

    vector<double> external_field; // h

    // J(i, j) = interaction between spins i & j
    // This matrix non-zero elements define
    // which spins in `initial_spins` are adjacent
    matrix<double> interaction;

    double temperature = 0.0; // T
    double magnetic_moment = 1.0; // mu
};

enum IsingModelParamsErrors {
    OK,
    INCONSISTENT_DIMENSIONS,
    BAD_XSIZE,
    NONZERO_INTERACTION_OF_NONADJACENT_SPINS,
    NEGATIVE_TEMPERATURE
};

class GeneralisedIsingModel {
protected:
    struct GeneralisedIsingParams params;

    vector<bool> spins; // current configuration

    unsigned int step_count = 0;

    // These static functions should really be methods of 'Params' class
    static IsingModelParamsErrors
    are_params_valid(struct GeneralisedIsingParams &params);

    static bool all_sizes_equal(GeneralisedIsingParams &params);

    static std::string get_error_message(IsingModelParamsErrors errors);

    double get_spin(size_t index) const;

    double get_spin_energy(size_t index) const;
public:
    explicit GeneralisedIsingModel(struct GeneralisedIsingParams initial_params);

    const vector<bool> &get_spins() const {
        return spins;
    }

    double get_total_energy() const;

    virtual void run(uint no_steps) = 0;

    void reset();

};

class TwoDimensionalIsingModel : public GeneralisedIsingModel {
    // Like Generalised IsingModel, but requires that J(i, j)
    // is non-zero only if spins[i] and spins[j] are adjacent when we treat
    // `spins` as a 2D matrix with dimensions row_size x col_size;
    // row_size x col_size must equal size

    explicit TwoDimensionalIsingModel(struct TwoDimensionalIsingParams params);
};

#endif //_ISING_SIMULATION_H_