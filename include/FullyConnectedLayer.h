#pragma once

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    class FullyConnectedLayer : public Layer {
    public:
        // Constructors
        FullyConnectedLayer(std::vector<ParameterMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * TODO
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons& input) noexcept override;
    };
}