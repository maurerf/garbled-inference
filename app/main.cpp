// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false
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
    const auto result = GarbledInference::NeuralNet::getInstance().inference({1,-2,3,-4});
    for(const auto& i : result)
        std::cout << i << std::endl;

    return 0;
}
