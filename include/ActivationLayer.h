#pragma once

namespace GarbledInference {

    /**
     * This class represents a single activation layer of a neural network.
     * It utilizes the Tiny Garble software interface to apply the garbled circuit protocol.
     *
     * What kind of activation function shall be applied is defined using a macro in Layer.h
     */
    class ActivationLayer : public Layer {
    public:
        // Constructors
        ActivationLayer(std::vector <Parameters> &weightMatrices, std::vector <LAYER_TYPE> &layerTypes) : Layer(
                weightMatrices, layerTypes) {}

        // Member functions
        /**
         * Overrides GarbledInference::Layer's member function.
         * Cf. documentation of base class.
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons &input) noexcept override;

        /**
         * Activation function used by activation layers
         *
         * @param i integer input of activation function
         * @return integer output of activation function
         */
        static inline double activation(const double &input) noexcept;
    };
}

