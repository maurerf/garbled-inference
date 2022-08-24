#pragma once

namespace GarbledInference {

    /**
     * This class represents a single convolutional layer of a neural network.
     *
     * The weights are expected to be in matrix form and per input data point.
     */
    class ConvolutionLayer : public Layer {
    public:
        // Constructors
        ConvolutionLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * Overrides GarbledInference::Layer's member function.
         * Cf. documentation of base class.
         * Note that stride = 1.
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons& input) noexcept override;
    };
}
