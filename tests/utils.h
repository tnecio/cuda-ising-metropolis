#ifndef ISING_UTILS_H
#define ISING_UTILS_H

#include <iostream>
#include <random>

#define TEST_SUITE_START(name) \
int name() { \
std::string test_suit_name = #name; \
int total_tests = 0; \
int res = 0;
int x_res;

#define TEST(x, ...) \
x_res = x(__VA_ARGS__); \
total_tests += 1; \
res += x_res

#define TEST_SUITE_END \
test_suite_end(test_suit_name, total_tests, res); \
return res; \
}

#define ASSERT(test_name, cond, msg) \
((cond) ? success(test_name) : fail(test_name, msg))

int fail(const std::string &name, const std::string &msg) {
    std::cout << name << " " << msg << "\033[1;31m FAIL\033[0m" << std::endl;
    return 1;
}

int success(const std::string &name) {
    std::cout << name << "\033[1;32m SUCCESS\033[0m" << std::endl;
    return 0;
}

void test_suite_end(const std::string &name, int total_tests, int res) {
    std::cout << "\033[1;34m" << name << ":\033[0m Performed " << total_tests << "; Success: "
              << total_tests - res << "; Failures: " << res << std::endl
              << std::endl;
}

double random_double(std::mt19937 random_engine) {
    std::uniform_real_distribution<> distribution(0., 1.);
    return distribution(random_engine);
}


#endif //ISING_UTILS_H
