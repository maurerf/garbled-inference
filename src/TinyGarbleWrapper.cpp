#include "TinyGarbleWrapper.h"


#include <utility>
#include <iostream>

namespace GarbledInference::Garbling {

    TinyGarbleWrapper &TinyGarbleWrapper::getInstance() {
        static TinyGarbleWrapper singleton {
            GarbledInference::Garbling::TinyGarbleWrapper("localhost", 1234, "/home/fdm/CLionProjects/garbled-inference/netlists/masked_relu64.scd")
        };
        return singleton;
    }

    TinyGarbleWrapper::TinyGarbleWrapper(std::string serverAddr, const int &serverPort, std::string scdFileLocation) :
            _port{serverPort},
            _connfd{-2}, // -2 = connection not yet open, -1 = connection failed
            _scd_file_address{std::move(scdFileLocation)},
            _server_ip{std::move(serverAddr)},
            _p_init_str{ReadFileOrPassHex("0")}, //unused
            _p_input_str{ReadFileOrPassHex("0")}, //unused
            _init_str{ReadFileOrPassHex("0")}, //unused
            //_input_f_hex_str is passed in evaluate()
            _terminate_period{0},
            _clock_cycles{1}, //todo: what exactly does this mean? what is 1 clock cycle
            _output_mask{0},  //todo: what exactly does this mean? does it tamper with output?
            _output_mode{OutputMode::consecutive},
            _disable_OT{false},
            _low_mem_foot{false} {

    }

    template<>
    std::string
    TinyGarbleWrapper::evaluate<GarbledInference::Garbling::ROLE::ALICE>(const std::string &input_f_hex_str) {

        string input_str = ReadFileOrPassHex(input_f_hex_str);
        std::string ans;

        if(_connfd == -2) {
            if ((_connfd = ServerInit(_port)) == -1) {
                std::cerr << "Cannot open the socket in port " << _port << std::endl;
            }
        }

        GarbleStr(_scd_file_address, _p_init_str, _p_input_str, _init_str,
                  input_str, _clock_cycles, _output_mask, _terminate_period,
                  _output_mode, _disable_OT, _low_mem_foot, &ans, _connfd);

        return ans;
    }


    template<>
    std::string TinyGarbleWrapper::evaluate<GarbledInference::Garbling::ROLE::BOB>(const std::string &input_f_hex_str) {

        string input_str = ReadFileOrPassHex(input_f_hex_str);
        std::string ans;

        if(_connfd == -2) {
            if ((_connfd = ClientInit(_server_ip.c_str(), _port)) == -1) {
                std::cerr << "Cannot connect to " << _server_ip << ":" << _port << std::endl;
            }
        }

        EvaluateStr(_scd_file_address, _p_init_str, _p_input_str, _init_str,
                    input_str, _clock_cycles, _output_mask, _terminate_period,
                    _output_mode, _disable_OT, _low_mem_foot, &ans, _connfd);

        return ans;
    }
}
