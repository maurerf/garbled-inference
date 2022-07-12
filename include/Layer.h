#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <variant>

//TODO: fix weak-vtables warning in clang
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

// define what activation function to use
#define GI_ACTIVATION_RELU

#define DEBUG_LAYERS
//#define DEBUG_LAYERS_VERBOSE

namespace GarbledInference {

    /**
    * TODO: Doxygen compliant interface comment.
    */
    typedef Eigen::MatrixXd ParameterMatrix;
    typedef std::vector<std::vector<std::variant<double, ParameterMatrix>>> Parameters; //size of inner vector = number of input feature maps. size of outer vector = number of output feature maps
    //TODO: most layer types (currently all but conv) use the same weight matrix for each input feature map. maybe change Parameters respecting that...
    typedef std::vector<Eigen::MatrixXd> Neurons; // size of vector = depth of input





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
        Layer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes);

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


        GarbledInference::Parameters _weights;
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
