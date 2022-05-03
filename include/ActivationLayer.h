#pragma once

#include "Layer.h"

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    class ActivationLayer : public Layer {
    public:
        // Constructors
        ActivationLayer(std::vector<WeightMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {};

        // Member functions
        /**
         * TODO
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        std::vector<int> process(std::vector<int> input) override;
    };
}