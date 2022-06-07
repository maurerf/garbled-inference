// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>

#include "NeuralNetConfig.h"
#include "NeuralNet.h"

/*
 * Main functions of Garbled Inference.
 */
int main() {
    // init dummy image (all zeros) and infer
    const auto result = GarbledInference::NeuralNet::getInstance().inference({Eigen::Matrix<double, 28,28>::Zero()});

    for(const auto& m : result) {
        std::cout << m << std::endl;
    }

    return 0;
}
