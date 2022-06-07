#pragma once

namespace GarbledInference {

    /**
     * TODO: Doxygen compliant interface comment.
     */
    class ConvolutionLayer : public Layer {
    public:
        // Constructors
        ConvolutionLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * TODO
         * Note that stride = 1.
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons& input) noexcept override;
    };
}
