#pragma once

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    template<size_t kernel_size, size_t stride>
    class MaxPoolingLayer : public Layer {
    public:
        // Constructors
        MaxPoolingLayer(std::vector<ParameterMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

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