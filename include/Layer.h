#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

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
         * @param layerTypes nonempty list of layer types
         * @return Layer object
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
        virtual std::vector<int> process(std::vector<int> input) = 0;

    private:
        //TODO: bias vector
        GarbledInference::WeightMatrix _weights;
        std::unique_ptr<Layer> _nextLayer;
    };
}

// Include Layer implementations
#include "FullyConnectedLayer.h"
#include "ActivationLayer.h"