#include <algorithm>
#include "Layer.h"

// Constructors
GarbledInference::Layer::Layer(std::vector<GarbledInference::WeightMatrix>& weightMatrices, std::vector<GarbledInference::LAYER_TYPE>& layerTypes)
: _weights(weightMatrices.front())
{
    // recursively init next layer
    weightMatrices.erase(weightMatrices.begin());
    layerTypes.erase(layerTypes.begin());

#ifdef DEBUG_LAYERS
    std::cout << "Constructed new layer! " << weightMatrices.size() << " layers remaining." << std::endl;
#endif

    if(!weightMatrices.empty() && !layerTypes.empty())
        switch (layerTypes.front()) {
            case LAYER_TYPE::FULLY_CONNECTED :
                _nextLayer = std::make_unique<GarbledInference::FullyConnectedLayer>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::ACTIVATION :
                _nextLayer = std::make_unique<GarbledInference::ActivationLayer>(weightMatrices, layerTypes); break;
        }
    else
        _nextLayer = nullptr;
}


// Member Functions
GarbledInference::NeuronVector GarbledInference::Layer::propagateForward(const GarbledInference::NeuronVector& input) noexcept {
#ifdef DEBUG_LAYERS
    std::cout << input << std::endl << std::endl;
#endif
    if(_nextLayer == nullptr) {
        return process(input);
    }
    else {
        return _nextLayer->propagateForward(process(input));
    }
}

constexpr double GarbledInference::ActivationLayer::activation(const double&  i) noexcept {
#ifdef GI_ACTIVATION_STEP
    return (i > 0) ? 1.0 : 0.0;
#endif
#ifdef GI_ACTIVATION_RELU
    return (i > 0) ? i : 0.0;
#endif

    //TODO: more afs
}

inline GarbledInference::NeuronVector GarbledInference::ActivationLayer::process(const GarbledInference::NeuronVector& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing activation layer!" << std::endl;
#endif

    return input.unaryExpr(&activation);
}

inline GarbledInference::NeuronVector GarbledInference::FullyConnectedLayer::process(const GarbledInference::NeuronVector& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing fully connected layer!" << std::endl;
#endif

    return input * _weights;
}

