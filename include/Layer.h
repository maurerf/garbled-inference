#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <variant>
#include <eigen3/Eigen/Eigen>
#include <TinyGarbleWrapper.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

// define what activation function to use
#define GI_ACTIVATION_RELU

#define ENABLE_ACTIVATION_LAYER_TIMING
#define DEBUG_LAYERS
//#define DEBUG_LAYERS_VERBOSE

namespace GarbledInference {

    /**
    * Basic data types used by the neural net implementation.
    */
    typedef Eigen::MatrixXd ParameterMatrix;
    typedef std::vector<std::vector<std::variant<double, ParameterMatrix>>> Parameters; //size of inner vector = number of input feature maps. size of outer vector = number of output feature maps
    typedef std::vector<Eigen::MatrixXd> Neurons; // size of vector = depth of input



    /**
    * Enum class encoding what kind of layer a GarbledInference::Layer instance represents.
    */
    enum class LAYER_TYPE {ACTIVATION, FULLY_CONNECTED, ADDITION, MAXPOOL_2, MAXPOOL_3, CONVOLUTION, RESHAPE};



    /**
     * Basic building block of the neural net.
     *
     * Base class defines recursive structure of chained Layer instances. Specialisations define individual behaviour.
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
         * @param input input to this layer's neurons
         * @return output of this layer's processing
         */
        inline virtual Neurons process(const Neurons & input) noexcept = 0;


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
#include "FlattenLayer.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
