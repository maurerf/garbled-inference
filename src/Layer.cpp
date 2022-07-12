#include <algorithm>
#include "Layer.h"

// Implementation of all constructors and member functions of GarbledInference::Layer and derived classes thereof.

// Constructors
GarbledInference::Layer::Layer(std::vector<GarbledInference::Parameters>& weightMatrices, std::vector<GarbledInference::LAYER_TYPE>& layerTypes)
: _weights(weightMatrices.front())
{
    // recursively init next layer
    weightMatrices.erase(weightMatrices.begin());
    layerTypes.erase(layerTypes.begin());

#ifdef DEBUG_LAYERS
    std::cout << "Constructed new layer! " << weightMatrices.size() << " layers remaining." << std::endl;
#endif

    if(!weightMatrices.empty() && !layerTypes.empty()) {
        switch (layerTypes.front()) {
            case LAYER_TYPE::ACTIVATION :
                _nextLayer = std::make_unique<GarbledInference::ActivationLayer>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::FULLY_CONNECTED :
                _nextLayer = std::make_unique<GarbledInference::FullyConnectedLayer>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::ADDITION :
                _nextLayer = std::make_unique<GarbledInference::AdditionLayer>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::MAXPOOL_2 :
                _nextLayer = std::make_unique<GarbledInference::MaxPoolingLayer<2, 2>>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::MAXPOOL_3 :
                _nextLayer = std::make_unique<GarbledInference::MaxPoolingLayer<3, 3>>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::CONVOLUTION :
                _nextLayer = std::make_unique<GarbledInference::ConvolutionLayer>(weightMatrices, layerTypes);
                break;
            case LAYER_TYPE::RESHAPE :
                _nextLayer = std::make_unique<GarbledInference::ReshapeLayer>(weightMatrices, layerTypes);
                break;
        }
    } else {
        _nextLayer = nullptr;
    }
}


// Member Functions
GarbledInference::Neurons GarbledInference::Layer::propagateForward(const GarbledInference::Neurons& input) noexcept {

#ifdef DEBUG_LAYERS_VERBOSE
    for(const auto& matrix : input) {
        std::cout << "\n" << matrix << std::endl;
    }
#endif

    if(_nextLayer == nullptr) {
#ifdef DEBUG_LAYERS
        std::cout << "PropagateForward(): Input size of final layer: " << input.size() << " x " << input.front().rows() << " x " << input.front().cols() << std::endl;
#endif
        return process(input);
    } else {
#ifdef DEBUG_LAYERS
        std::cout << "PropagateForward(): Input size: " << input.size() << " x " << input.front().rows() << " x " << input.front().cols() << std::endl;
#endif
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
    //more activation functions here
}

inline GarbledInference::Neurons GarbledInference::ActivationLayer::process(const GarbledInference::Neurons& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing activation layer!" << std::endl;
#endif

    Neurons output;

    for(const auto& d : input) {
        output.emplace_back(d.unaryExpr(&activation));
    }

    return output;
}

inline GarbledInference::Neurons GarbledInference::FullyConnectedLayer::process(const GarbledInference::Neurons& input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing fully connected layer!" << std::endl;
#endif

    Neurons output;




    for(size_t d = 0; d < input.size(); d++) {
        // todo: assert: _weights[d].size == 1
        const auto w = _weights[d].front();
        if (std::holds_alternative<double>(w)) {
            output.emplace_back(input[d] * std::get<double>(w));
        } else if (std::holds_alternative<ParameterMatrix>(w)) {
            output.emplace_back(std::get<ParameterMatrix>(w) * input[d]);
        }
    }

    return output;
}

inline GarbledInference::Neurons GarbledInference::AdditionLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing addition layer!" << std::endl;
#endif

    Neurons output;

    for(size_t d = 0; d < input.size(); d++) {
        // todo: assert: _weights[d].size == 1
        const auto w = _weights[d].front();
        if (std::holds_alternative<double>(w)) {

            /*
             * Eigen3 does not have scalar addition for matrices...
             *
             * https://stackoverflow.com/questions/65455597/adding-scalar-to-eigen-matrix-vector
             */
            const ParameterMatrix addend = ParameterMatrix::Ones(input[d].rows(), input[d].cols()) * std::get<double>(w);

            output.emplace_back(input[d] + addend);
            continue;
        }
        if (std::holds_alternative<ParameterMatrix>(w)) {
            output.emplace_back(input[d] + std::get<ParameterMatrix>(w));
        }
    }

    return output;
}

template<Eigen::Index kernel_size, Eigen::Index stride>
inline GarbledInference::Neurons GarbledInference::MaxPoolingLayer<kernel_size, stride>::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing max-pooling layer!" << std::endl;
#endif

    Neurons output = Neurons(input.size());

    // process all input feature maps iteratively
    for(size_t d = 0; d < input.size(); d++) {

        const auto inputWidth = input[d].rows();
        const auto inputHeight = input[d].cols();

        // sanity check
        if(stride > inputWidth or stride > inputHeight) {
            std::cout << "Warning: Processing max pooling layer with stride greater than input size." << std::endl;
        }

        // resize output for this feature map TODO: is this correct?
        output[d].resize((inputWidth) / stride, (inputHeight) / stride);

        // TODO: more efficient resizing
        for (Eigen::Index x = 0; x < inputWidth - (kernel_size - 1); x += stride) {
            for (Eigen::Index y = 0; y < inputHeight - (kernel_size - 1); y += stride) {
                auto poolBlock = input[d].block(x, y, kernel_size, kernel_size);
                output[d](x / stride, y / stride) = poolBlock.maxCoeff();
            }
        }
    }
    return output;
}


inline GarbledInference::Neurons GarbledInference::ConvolutionLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing convolution layer!" << std::endl;
#endif

    // init output
    Neurons output = Neurons(_weights.size());

    // handle all feature maps of output iteratively (f)
    for(size_t f = 0; f < _weights.size(); f++) {
        // resize output feature map
        output[f].resize(input.front().rows(), input.front().cols());

        // for each input... (d)
        for (size_t d = 0; d < input.size(); d++) {

            // unbind std::variant monad for depth = d. Entries of _weights are assumed to be a matrix for convolutional layers TODO: check wMatrix/ exception
            const auto wMatrix = std::get<ParameterMatrix>(_weights[f][d]);

            const auto inputWidth = input[d].rows();
            const auto inputHeight = input[d].cols();
            const auto kernelWidth = wMatrix.rows();
            const auto kernelHeight = wMatrix.cols();

            // for each pixel... (x,y)
            for (auto x = 0; x < inputWidth; x++) {
                for (auto y = 0; y < inputHeight; y++) {
                    //...init pixel with zero...
                    output[f](x, y) = 0.0;

                    //...then iterate along its kernel (k_x, k_y)
                    for (auto k_x = 0; k_x < kernelWidth; k_x++) {
                        for (auto k_y = 0; k_y < kernelHeight; k_y++) {
                            //boundary handling ("zero-padding"):
                            if (
                                    x > kernelWidth / 2 and x < inputWidth - kernelWidth / 2
                                    and y > kernelHeight / 2 and y < inputHeight - kernelHeight / 2
                            ) {
                                // ... and add result of convolution to current feature map (f)
                                output[f](x, y) +=
                                        // offset of (0,0) of kernel relative to pixel is half of kernel size, rounded down TODO verify correctness
                                        (input[d](x + k_x - kernelWidth / 2, y + k_y - kernelHeight / 2)
                                         *
                                         wMatrix(k_x, k_y));
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}


//TODO: this is a bit of a mess and probably doesnt work the same way ONNX runtime reshapes
inline GarbledInference::Neurons GarbledInference::ReshapeLayer::process(const GarbledInference::Neurons &input) noexcept {

#ifdef DEBUG_LAYERS
    std::cout << "Processing reshape layer!" << std::endl;
#endif

    // create 1d "matrix" to append all neurons to
    Eigen::MatrixXd list = {};
    list.resize(static_cast<long>(input.size()) * input.front().rows() * input.front().cols(), 1);
    long c = 0; // entry counter for list

    // append all entries from all feature maps to 1d vector
    for(const auto & d : input) {
        for(auto x = 0; x < d.rows(); x++) {
            for(auto y = 0; y < d.cols(); y++) {
                list(c,0) = d(x,y);
                c++;
            }
        }
    }

    // unbind std::variant monad for depth = d. Entries of _weights is assumed to be a scalar for reshape layers TODO: check w/ exception
    const auto x_size = static_cast<size_t>(std::get<double>(_weights[0].front()));
    const auto y_size = static_cast<size_t>(std::get<double>(_weights[1].front()));

    // reshape 1d vector to desired shape, contain it in a std::vector
    return { list.reshaped<Eigen::AutoOrder>(x_size, y_size) };
}
