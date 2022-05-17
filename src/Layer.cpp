#include <algorithm>
#include "Layer.h"

// Constructors
GarbledInference::Layer::Layer(std::vector<GarbledInference::ParameterMatrix>& weightMatrices, std::vector<GarbledInference::LAYER_TYPE>& layerTypes)
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
            case LAYER_TYPE::ACTIVATION :
                _nextLayer = std::make_unique<GarbledInference::ActivationLayer>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::FULLY_CONNECTED :
                _nextLayer = std::make_unique<GarbledInference::FullyConnectedLayer>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::ADDITION :
                _nextLayer = std::make_unique<GarbledInference::AdditionLayer>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::MAXPOOL_2 :
                _nextLayer = std::make_unique<GarbledInference::MaxPoolingLayer<2,2>>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::MAXPOOL_3 :
                _nextLayer = std::make_unique<GarbledInference::MaxPoolingLayer<3,3>>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::CONVOLUTION :
                _nextLayer = std::make_unique<GarbledInference::ConvolutionLayer>(weightMatrices, layerTypes); break;
            case LAYER_TYPE::RESHAPE :
                _nextLayer = std::make_unique<GarbledInference::ReshapeLayer>(weightMatrices, layerTypes); break;
        }
    else
        _nextLayer = nullptr;
}


// Member Functions
GarbledInference::Neurons GarbledInference::Layer::propagateForward(const GarbledInference::Neurons& input) noexcept {
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

inline GarbledInference::Neurons GarbledInference::ActivationLayer::process(const GarbledInference::Neurons& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing activation layer!" << std::endl;
#endif

    return input.unaryExpr(&activation);
}

inline GarbledInference::Neurons GarbledInference::FullyConnectedLayer::process(const GarbledInference::Neurons& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing fully connected layer!" << std::endl;
#endif

    return input * _weights;
}

inline GarbledInference::Neurons GarbledInference::AdditionLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing max-pooling layer!" << std::endl;
#endif

    return input + _weights;
}

template<size_t kernel_size, size_t stride>
inline GarbledInference::Neurons GarbledInference::MaxPoolingLayer<kernel_size, stride>::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing max-pooling layer!" << std::endl;
#endif

    Neurons output;
    //TODO for n dims
    for(size_t x = 0; x < static_cast<size_t>(input.rows()); x += stride) {
        for(size_t y = 0; y < static_cast<size_t>(input.cols()); y += stride) {
            const auto poolBlock = input.block(x,y, x + kernel_size - 1, y + kernel_size - 1);
            output(x/stride, y/stride) = poolBlock.maxCoeff();
        }
    }
    return output;
}

inline GarbledInference::Neurons GarbledInference::ConvolutionLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing addition layer!" << std::endl;
#endif
    GarbledInference::Neurons output;

    //TODO: impl for more dims / depth

    const auto inputHeight = input.rows();
    const auto inputWidth = input.cols();
    const auto kernelWidth = _weights.rows();
    const auto kernelHeight = _weights.cols();

    // for each pixel...
    for(auto x = 0; x < inputWidth; x++) {
        for(auto y = 0; y < inputHeight; y++) {
            //...iterate along its kernel
            for(auto k_x = 0; k_x < kernelWidth; k_x++) {
                for(auto k_y = 0; k_y < kernelHeight; k_y++) {
                    if(x > kernelWidth/2 and y > kernelHeight/2) {//boundary handling
                        // offset of (0,0) of kernel relative to pixel is half of kernel size, rounded down TODO: verify these divisions are rounded down
                        output(x, y) *= ( input(x + k_x - kernelWidth/2, y + k_y - kernelHeight/2) * _weights(k_x, k_y));
                    }
                }
            }
        }
    }


    //TODO
    return input;
}

inline GarbledInference::Neurons GarbledInference::ReshapeLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing reshape layer!" << std::endl;
#endif

    //TODO allow reshape to m dimensions (currently : n->2)
    Neurons output (input);
    output.reshaped<Eigen::AutoOrder>(_weights(0,0), _weights(0,1));
    return output;
}