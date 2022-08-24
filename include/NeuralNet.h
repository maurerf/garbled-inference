#pragma once

#include <vector>
#include "Layer.h"
#include "TinyGarbleWrapper.h"

namespace GarbledInference {

    /**
     * Represents the entire neural net. Meant to be used as a singleton.
     *
     * Manages instances of GarbledInference::Layer.
    */
    class NeuralNet {
    public:
        // Constructors
        /**
         * Access to the singleton NN.
         */
        static NeuralNet &getInstance();

        NeuralNet(const NeuralNet&) = delete;

        NeuralNet(NeuralNet&&) = delete;

        //Member functions

        /**
         * Infers features from an input image.
         *
         * @param input data to be classified
         * @return mapped feature vector
         */
        Eigen::Index inference(const Neurons& input) noexcept;

        NeuralNet& operator=(const NeuralNet&) = delete;

        NeuralNet& operator=(NeuralNet&&) = delete;

    private:
        NeuralNet();

        std::unique_ptr<GarbledInference::Layer> _firstLayer;
    };
}
