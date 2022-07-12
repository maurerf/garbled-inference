#pragma once

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    class ActivationLayer : public Layer {
    public:
        // Constructors
        ActivationLayer(std::vector <Parameters> &weightMatrices, std::vector <LAYER_TYPE> &layerTypes) : Layer(
                weightMatrices, layerTypes) {}

        // Member functions
        /**
         * TODO
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons &input) noexcept override;

        /**
         * Activation function used by activation layers
         * @param i integer input of activation function
         * @return integer output of activation function
         */
        static constexpr double activation(const double &i) noexcept;
    };
}

