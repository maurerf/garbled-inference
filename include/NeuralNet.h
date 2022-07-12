#pragma once

#include <vector>
#include "Layer.h"

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    class NeuralNet {
    public:
        // Constructors
        /**
         * Creates a new neural net.
         */
        static NeuralNet& getInstance();

        NeuralNet(const NeuralNet&) = delete;

        NeuralNet(NeuralNet&&) = delete;

        //Member functions

        /**
         * Infers features from an input vector.
         *
         * @param input data to be classified
         * @return mapped feature vector
         */
        Neurons inference(const Neurons& input) noexcept;

        NeuralNet& operator=(const NeuralNet&) = delete;

        NeuralNet& operator=(NeuralNet&&) = delete;

    private:
        NeuralNet();

        std::unique_ptr<GarbledInference::Layer> _firstLayer;
    };
}
