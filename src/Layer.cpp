#include "Layer.h"

// Constructors
GarbledInference::Layer::Layer(std::vector<GarbledInference::WeightMatrix>& weightMatrices, std::vector<GarbledInference::LAYER_TYPE>& layerTypes)
: _weights(weightMatrices.front())
{
    // recursively init next layer
    weightMatrices.erase(weightMatrices.begin());
    layerTypes.erase(layerTypes.begin());

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
std::vector<int> GarbledInference::Layer::propagateForward(std::vector<int> input) noexcept {
    if(_nextLayer == nullptr)
        return process(input);
    else
        return _nextLayer->propagateForward(process(input));
}
