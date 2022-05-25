#pragma once

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    class ReshapeLayer : public Layer {
    public:
        // Constructors
        ReshapeLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

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
