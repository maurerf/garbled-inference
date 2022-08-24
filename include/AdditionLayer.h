#pragma once

namespace GarbledInference {

    /**
     * This class represents a single addition layer of a neural network.
     *
     * The input weights of the constructor are expected to be either per feature map or per input data point individually.
     */
    class AdditionLayer : public Layer {
    public:
        // Constructors
        AdditionLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * Overrides GarbledInference::Layer's member function.
         * Cf. documentation of base class.
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons& input) noexcept override;
    };
}
