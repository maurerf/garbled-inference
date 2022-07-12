#include "NeuralNet.h"

// include weights of model
#include "weights/MNIST/convolution_1.h.in"
#include "weights/MNIST/convolution_2.h.in"
#include "weights/MNIST/dense_1.h.in"
#include "weights/MNIST/addition_1.h.in"
#include "weights/MNIST/addition_2.h.in"
#include "weights/MNIST/addition_3.h.in"
#include "weights/MNIST/reshape_1.h.in"


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
    GarbledInference::Parameters reshape1_w {
            {
                256.0
            },
            {
                1.0
            }
    };

    //TODO: const correctness for these two?

    std::vector<GarbledInference::Parameters> MNIST_weights {
            Weights::MNIST::CONVOLUTION_1,
            Weights::MNIST::ADD_1,
            {/*RELU*/},
            {/*Max-Pooling*/},
            Weights::MNIST::CONVOLUTION_2,
            Weights::MNIST::ADD_2,
            {/*RELU*/},
            {/*Max-Pooling*/},
            Weights::MNIST::RESHAPE_1,
            Weights::MNIST::DENSE_1,
            Weights::MNIST::ADD_3,
    };

    //define layer topology
    std::vector<GarbledInference::LAYER_TYPE> MNIST_layers {
            LAYER_TYPE::CONVOLUTION,
            LAYER_TYPE::ADDITION,
            LAYER_TYPE::ACTIVATION,
            LAYER_TYPE::MAXPOOL_2,
            LAYER_TYPE::CONVOLUTION,
            LAYER_TYPE::ADDITION,
            LAYER_TYPE::ACTIVATION,
            LAYER_TYPE::MAXPOOL_3,
            LAYER_TYPE::RESHAPE,
            LAYER_TYPE::FULLY_CONNECTED,
            LAYER_TYPE::ADDITION
    };

    // init first layer of net manually
    _firstLayer = std::make_unique<GarbledInference::ConvolutionLayer>(MNIST_weights, MNIST_layers);
}

GarbledInference::Neurons GarbledInference::NeuralNet::inference(const GarbledInference::Neurons& input) noexcept {
    return _firstLayer->propagateForward(input);
}
