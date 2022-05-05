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
         * Creates a new neural net. TODO: input parametrisation
         */
        static NeuralNet& getInstance() {
            [[clang::no_destroy]] static NeuralNet singleton {}; //TODO: check: is this a memory leak?
            return singleton;
        }

        NeuralNet(const NeuralNet&) = delete;

        NeuralNet(NeuralNet&&) = delete;

        //Member functions

        /**
         * Infers features from an input vector.
         *
         * @param input data to be classified
         * @return mapped feature vector
         */
        std::vector<int> inference(std::vector<int> input) noexcept;

        NeuralNet& operator=(const NeuralNet&) = delete;

        NeuralNet& operator=(NeuralNet&&) = delete;

    private:
        NeuralNet();

        std::unique_ptr<GarbledInference::Layer> _firstLayer;
    };
}
