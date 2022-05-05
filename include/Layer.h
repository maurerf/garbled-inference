#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

//TODO: fix this properly
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

// define what activation function to use
#define GI_ACTIVATION_STEP
#define DEBUG_LAYERS

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    typedef std::vector<std::vector<double>> WeightMatrix;





    /**
    * TODO: Doxygen compliant interface comment.
    */
    enum class LAYER_TYPE {FULLY_CONNECTED, ACTIVATION};





    /**
    * TODO: Doxygen compliant interface comment.
    */
    class Layer {
    public:
        // Constructors
        /**
         * Constructs a neural network layer.
         *
         * @param weightMatrices nonempty list of layer weight matrices
         * @param layerTypes nonempty list of layer type
         */
        Layer(std::vector<WeightMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes);

        Layer() = delete;

        Layer(const Layer&) = delete;

        Layer(Layer&&) = delete;

        // Destructor
        virtual ~Layer() = default;

        //Member functions
        /**
         * Implements forward propagation beginning in this layer.
         *
         * @param input input vector to be classified
         * @return feature vector
         */
        std::vector<int> propagateForward(std::vector<int> input) noexcept;

        Layer& operator=(const Layer&) = delete;

        Layer& operator=(Layer&&) = delete;

    protected:

        /**
         * Virtual members function containing this layer's i/o mapping.
         *
         *
         * @param input input to this layer's neurons
         * @return output of this layer's processing
         */
        inline virtual std::vector<int> process(const std::vector<int>& input) noexcept { return input; }

    private:
        //TODO: bias vector
        GarbledInference::WeightMatrix _weights;
        std::unique_ptr<Layer> _nextLayer;
    };





    /**
    * TODO: Doxygen compliant interface comment.
    */
    class ActivationLayer : public Layer {
    public:
        // Constructors
        ActivationLayer(std::vector <WeightMatrix> &weightMatrices, std::vector <LAYER_TYPE> &layerTypes) : Layer(
                weightMatrices, layerTypes) {}

        // Member functions
        /**
         * TODO
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        std::vector<int> process(const std::vector<int> &input) noexcept override;

        /**
         * Activation function used by activation layers
         * @param i integer input of activation function
         * @return integer output of activation function
         */
        static constexpr int activation(const int &i) noexcept;
    };





    /**
    * TODO: Doxygen compliant interface comment.
    */
    class FullyConnectedLayer : public Layer {
    public:
        // Constructors
        FullyConnectedLayer(std::vector<WeightMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * TODO
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        std::vector<int> process(const std::vector<int>& input) noexcept override;
    };
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
