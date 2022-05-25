#pragma once

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    template<Eigen::Index kernel_size, Eigen::Index stride>
    class MaxPoolingLayer : public Layer {
    public:
        // Constructors
        MaxPoolingLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

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
