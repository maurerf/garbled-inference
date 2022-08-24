#pragma once

namespace GarbledInference {

    /**
     * This class represents a single dense/fully-connected layer of a neural network.
     *
     * The weights are expected to be in matrix form and per input data point.
    */
    class FullyConnectedLayer : public Layer {
    public:
        // Constructors
        FullyConnectedLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

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
