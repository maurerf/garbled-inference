#pragma once

namespace GarbledInference {

    /**
     * This class represents a single max-pooling layer of a neural network.
     *
     * The kernel size and stride of the pooling aggregation are set during runtime via templates. It does not use weights.
     */
    template<Eigen::Index kernel_size, Eigen::Index stride>
    class MaxPoolingLayer : public Layer {
    public:
        // Constructors
        MaxPoolingLayer(std::vector<Parameters>& weightMatrices, std::vector<LAYER_TYPE>& layerTypes) : Layer(weightMatrices, layerTypes) {}

        // Member functions
        /**
         * Overrides GarbledInference::Layer's member function.
         * Cf. documentation of base class.
         *
         * @param input see GarbledInference::Layer::process
         * @return see GarbledInference::Layer::process
         */
        Neurons process(const Neurons& input) noexcept override;
    };

    // explicit forward declarations of specialisations
    //template<>
    //Neurons MaxPoolingLayer<2, 2>::process(const Neurons &input) noexcept;

    //template<>
    //Neurons MaxPoolingLayer<3, 3>::process(const Neurons &input) noexcept;
}

