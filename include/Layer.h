#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <eigen3/Eigen/Eigen>

//TODO: fix weak-vtables warning in clang
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

// define what activation function to use
#define GI_ACTIVATION_RELU
//#define DEBUG_LAYERS

namespace GarbledInference {

    /**
     * TODO
     *  replicate mnist in NeuralNet/main
     *  change file structure
     *
     */



    /**
    * TODO: Doxygen compliant interface comment.
    */
    typedef Eigen::MatrixXd ParameterMatrix;
    typedef /*Eigen::RowVectorXd*/ Eigen::MatrixXd Neurons; //TODO: vector or matrix? ask tatjana





    /**
    * TODO: Doxygen compliant interface comment.
    */
    enum class LAYER_TYPE {ACTIVATION, FULLY_CONNECTED, ADDITION, MAXPOOL_2, MAXPOOL_3, CONVOLUTION, RESHAPE};





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
        Layer(std::vector<ParameterMatrix>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes);

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
        Neurons propagateForward(const Neurons& input) noexcept;

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
        inline virtual Neurons process(const Neurons & input) noexcept { return input; } //TODO: make pure virtual


        GarbledInference::ParameterMatrix _weights;
        //TODO: bias vector? ask tatjana
    private:
        std::unique_ptr<Layer> _nextLayer;
    };
}

// include derived classes
#include "ActivationLayer.h"
#include "FullyConnectedLayer.h"
#include "AdditionLayer.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "ReshapeLayer.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
