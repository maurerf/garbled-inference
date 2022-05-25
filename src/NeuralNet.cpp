#include "NeuralNet.h"

GarbledInference::NeuralNet &GarbledInference::NeuralNet::getInstance() {
    {
#ifdef __clang__
        [[clang::no_destroy]]
#endif
        static NeuralNet singleton {}; //TODO: check: is this a memory leak?
        return singleton;
    }
}

/*
  * cf. pre-trained ONNX model of MNIST
  *
  * TODO: move all of this config to .in
  */
GarbledInference::NeuralNet::NeuralNet() {
    //define layer topology
    std::vector<GarbledInference::LAYER_TYPE> MNIST_layers {
        LAYER_TYPE::CONVOLUTION,
        LAYER_TYPE::ADDITION,
        LAYER_TYPE::ACTIVATION,
        LAYER_TYPE::MAXPOOL_2,
        LAYER_TYPE::CONVOLUTION,
        LAYER_TYPE::ADDITION,
        LAYER_TYPE::ACTIVATION,
        LAYER_TYPE::MAXPOOL_3,
        LAYER_TYPE::RESHAPE,
        LAYER_TYPE::FULLY_CONNECTED,
        LAYER_TYPE::ADDITION
    };

    // define weight matrices (for fully connected layers)
    const GarbledInference::Parameters conv1_w {
            Eigen::MatrixXd{ // d = 0
                    {
                            -0.008905669674277306,
                            -0.23690743744373322,
                            -0.5088216662406921,
                            -0.06456177681684494,
                            0.14181184768676758
                    },
                    {
                            -0.5919761657714844,
                            -0.4752853810787201,
                            -0.049348145723342896,
                            0.7682155966758728,
                            0.26346519589424133
                    },
                    {
                            -0.4917634427547455,
                            0.05561765283346176,
                            1.0189645290374756,
                            0.5547041893005371,
                            -0.4416643977165222
                    },
                    {
                            -0.15953698754310608,
                            0.5575414896011353,
                            0.5920912623405457,
                            -0.2947413921356201,
                            -0.6131798028945923
                    },
                    {
                            0.03849884867668152,
                            0.22601930797100067,
                            -0.21855629980564117,
                            -0.47719430923461914,
                            -0.2917049527168274
                    }
            },
            Eigen::MatrixXd{ // d = 1
                    {
                            -0.039882346987724304,
                            0.2185998111963272,
                            0.49872705340385437,
                            0.42367178201675415,
                            0.048547789454460144
                    },
                    {
                            -0.10118173062801361,
                            -0.2769094705581665,
                            -0.004399026278406382,
                            0.5271722078323364,
                            0.40340957045555115
                    },
                    {
                            -0.15083789825439453,
                            -0.32669755816459656,
                            -0.15898573398590088,
                            0.35899585485458374,
                            0.3726355731487274
                    },
                    {
                            -0.5676766633987427,
                            -0.38571643829345703,
                            -0.1736956238746643,
                            0.22803734242916107,
                            0.33274900913238525
                    },
                    {
                            -0.38078179955482483,
                            -0.2294795960187912,
                            -0.06623970717191696,
                            -0.01456076093018055,
                            0.2849120795726776
                    }
            },
            Eigen::MatrixXd{ // d = 2
                    {
                            -0.023685239255428314,
                            0.10450534522533417,
                            0.253627210855484,
                            0.3617715537548065,
                            0.6909624934196472
                    },
                    {
                            0.33491653203964233,
                            0.4169996976852417,
                            0.38482844829559326,
                            0.3294100761413574,
                            0.24301494657993317
                    },
                    {
                            -0.469070702791214,
                            -0.07595888525247574,
                            -0.1445249766111374,
                            -0.17356260120868683,
                            -0.24677062034606934
                    },
                    {
                            -0.9726812243461609,
                            -0.7156173586845398,
                            -0.5382127165794373,
                            -0.6167798638343811,
                            -0.44845151901245117
                    },
                    {
                            -0.19898280501365662,
                            -0.3333558738231659,
                            -0.3490765392780304,
                            -0.14603577554225922,
                            -0.06359775364398956
                    }
            },
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 3
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 4
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 5
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 6
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ } // d = 7
    };
    GarbledInference::Parameters add1_w {
            -0.08224882185459137,
            -0.10886877775192261,
            -0.14103959500789642,
            -0.20486916601657867,
            -0.17913565039634705,
            -0.2154383808374405,
            -0.1338050663471222,
            -0.19572456181049347,
            -0.26825064420700073,
            -0.25821220874786377,
            -0.07615606486797333,
            0.01328414585441351,
            -0.004444644320756197,
            -0.41474083065986633,
            -0.17879115045070648,
            -0.03865588828921318,
    };
    GarbledInference::Parameters conv2_w {
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 0
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 1
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 2
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 3
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 4
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 5
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 6
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 7
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 8
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 9
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 10
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 11
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 12
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 13
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ }, // d = 14
            Eigen::MatrixXd { /* TODO: paste weights from onnx model */ } // d = 15
    };
    GarbledInference::Parameters add2_w {
            -0.08224882185459137,
            -0.10886877775192261,
            -0.14103959500789642,
            -0.20486916601657867,
            -0.17913565039634705,
            -0.2154383808374405,
            -0.1338050663471222,
            -0.19572456181049347,
            -0.26825064420700073,
            -0.25821220874786377,
            -0.07615606486797333,
            0.01328414585441351,
            -0.004444644320756197,
            -0.41474083065986633,
            -0.17879115045070648,
            -0.03865588828921318
    };
    GarbledInference::Parameters reshape1_w {
        256.0,
        1.0
    };
    GarbledInference::Parameters fc1_w {
        Eigen::MatrixXd {
            /*TODO*/
        }
    };
    GarbledInference::Parameters add3_w {
            Eigen::MatrixXd { //d = 0
                    {
                        -0.04485602676868439,
                         0.007791661191731691,
                         0.06810081750154495,
                         0.02999374084174633,
                         -0.1264096349477768,
                         0.14021874964237213,
                         -0.055284902453422546,
                         -0.04938381537795067,
                         0.08432205021381378,
                         -0.05454041436314583
                    }
            }
    };



    std::vector<GarbledInference::Parameters> MNIST_weights {
            conv1_w,
            add1_w,
            {/*RELU*/},
            {/*Max-Pooling*/},
            conv2_w,
            add2_w,
            {/*RELU*/},
            {/*Max-Pooling*/},
            reshape1_w,
            fc1_w,
            add3_w
    };

//TODO: fixme
    if(MNIST_layers.front() == LAYER_TYPE::FULLY_CONNECTED) {
        _firstLayer = std::make_unique<GarbledInference::FullyConnectedLayer>(MNIST_weights, MNIST_layers);
    }
    else {
        _firstLayer = std::make_unique<GarbledInference::ActivationLayer>(MNIST_weights, MNIST_layers);
    }

}

GarbledInference::Neurons GarbledInference::NeuralNet::inference(const GarbledInference::Neurons& input) noexcept {
    return _firstLayer->propagateForward(input);
}
