#include "NeuralNet.h"

GarbledInference::NeuralNet::NeuralNet() {
    //define layer topology
    std::vector<GarbledInference::LAYER_TYPE> layers = {
            LAYER_TYPE::FULLY_CONNECTED,
            LAYER_TYPE::ACTIVATION,
            LAYER_TYPE::FULLY_CONNECTED,
            LAYER_TYPE::ACTIVATION
    };

    // define weight matrices (for fully connected layers)
    const GarbledInference::WeightMatrix fc1_weights = {
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0}
    };


    const GarbledInference::WeightMatrix fc2_weights = {
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0}
    };


    std::vector<GarbledInference::WeightMatrix> weights = {
            fc1_weights,
            {/*activation layer*/},
            fc2_weights,
            {/*activation layer*/}
    };


    if(layers.front() == LAYER_TYPE::FULLY_CONNECTED) {
        _firstLayer = std::make_unique<GarbledInference::FullyConnectedLayer>(weights, layers);
    }
    else {
        _firstLayer = std::make_unique<GarbledInference::ActivationLayer>(weights, layers);
    }

}

std::vector<int> GarbledInference::NeuralNet::inference(std::vector<int> input) noexcept {
    return _firstLayer->propagateForward(input);
}
