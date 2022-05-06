#include "NeuralNet.h"

GarbledInference::NeuralNet &GarbledInference::NeuralNet::getInstance() {
    {
#ifdef __clang__
        [[clang::no_destroy]]
#endif
        static NeuralNet singleton {}; //TODO: check: is this a memory leak?
        return singleton;
    }
}

GarbledInference::NeuralNet::NeuralNet() {
    //define layer topology
    std::vector<GarbledInference::LAYER_TYPE> layers {
            LAYER_TYPE::FULLY_CONNECTED,
            LAYER_TYPE::ACTIVATION,
            LAYER_TYPE::FULLY_CONNECTED,
            LAYER_TYPE::ACTIVATION
    };

    // define weight matrices (for fully connected layers)
    const GarbledInference::WeightMatrix fc1_weights {
            {3.0, 1.0, 1.3, -1.0},
            {-1.0, -5.0, 1.0, 1.0},
            {1.0, 4.0, 1.0, -1.55},
            {-12.0, 1.0, 10.0, 1.0}
    };


    const GarbledInference::WeightMatrix fc2_weights {
            {-10.0, 1.0, 15.5, 1.0},
            {-33.0, 1.0, 1.0, 1.3},
            {1.0, 1.0, 0.05, -1.0},
            {-11.5, 1.0, 1.0, 1.0}
    };


    std::vector<GarbledInference::WeightMatrix> weights {
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

GarbledInference::NeuronVector GarbledInference::NeuralNet::inference(const GarbledInference::NeuronVector& input) noexcept {
    return _firstLayer->propagateForward(input);
}
