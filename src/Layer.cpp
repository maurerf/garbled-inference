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
std::vector<int> GarbledInference::Layer::propagateForward(std::vector<int> input) noexcept {
    if(_nextLayer == nullptr) {
        return process(input);
    }
    else {
        return _nextLayer->propagateForward(process(input));
    }
}

constexpr int GarbledInference::ActivationLayer::activation(const int&  i) noexcept {
#ifdef GI_ACTIVATION_STEP
    return static_cast<int>(i > 0);
#elifdef GI_ACTIVATION_RELU
    return (i > 0) ? i : 0;
#else
    static_assert(false, "No activation function defined for GI::ActivationLayer!");
#endif
    //TODO: more afs
}

inline std::vector<int> GarbledInference::ActivationLayer::process(const std::vector<int>& input) noexcept {
    //TODO: redesign this as a function, not a loop
    std::vector<int> ans;
    for(const auto& i : input) {
        ans.emplace_back(activation(i));
    }

#ifdef DEBUG_LAYERS
    std::cout << "Processed activation layer!" << std::endl;
#endif

    return ans;
}

inline std::vector<int> GarbledInference::FullyConnectedLayer::process(const std::vector<int>& input) noexcept {
    //TODO

#ifdef DEBUG_LAYERS
    std::cout << "Processed fully connected layer!" << std::endl;
#endif

    return input;
}

