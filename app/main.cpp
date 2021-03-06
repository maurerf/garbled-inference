// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <array>
#include <boost/asio.hpp>

#include "NeuralNetConfig.h"
#include "NeuralNet.h"
#include "Client.h"


/*
 * Main function of Garbled Inference user application.
 */
int main() {
    // create client & connect to server
    /*using namespace GarbledInference::Networking;
    [[maybe_unused]] Client client {
        [](const std::string& message) {
            std::cout << "Client: New message: " << message << std::endl;
        }
    };

    client.start();

    std::cout << client.get();*/

    // init dummy image (all zeros) and infer (TODO read from file)
    GarbledInference::Neurons input = {Eigen::Matrix<double, 28,28>::Zero()};

    // make some pixels white
    for(Eigen::Index row = 0; row < input[0].rows(); row++) {
        input[0](row, 13) = 1.0;
        input[0](row, 14) = 1.0;
        input[0](row, 15) = 1.0;
    }

    for(const auto& m : input) {
        std::cout << m << std::endl;
    }

    const auto result = GarbledInference::NeuralNet::getInstance().inference(input);

    for(const auto& m : result) {
        std::cout << m << std::endl;
    }

    return 0;
}


