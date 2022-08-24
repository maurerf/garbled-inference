#pragma once

namespace GarbledInference {

    /**
    * This class represents a single flatten/reshape layer of a neural network.
    */
    class FlattenLayer : public Layer {
    public:
        // Constructors
        FlattenLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

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
