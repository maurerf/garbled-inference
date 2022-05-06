#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <eigen3/Eigen/Eigen>

//TODO: fix this properly
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

// define what activation function to use
#define GI_ACTIVATION_RELU
//#define DEBUG_LAYERS

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    typedef Eigen::MatrixXd WeightMatrix;
    typedef Eigen::RowVectorXd NeuronVector; //TODO: better name suited for first and last layer as well





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
        NeuronVector propagateForward(const NeuronVector& input) noexcept;

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
        inline virtual NeuronVector process(const NeuronVector & input) noexcept { return input; } //TODO: make pure virtual


        GarbledInference::WeightMatrix _weights;
        //TODO: bias vector
    private:
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
        NeuronVector process(const NeuronVector &input) noexcept override;

        /**
         * Activation function used by activation layers
         * @param i integer input of activation function
         * @return integer output of activation function
         */
        static constexpr double activation(const double &i) noexcept;
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
        NeuronVector process(const NeuronVector& input) noexcept override;
    };
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
