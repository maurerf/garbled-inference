#pragma once

#include <string>
#include <garbled_circuit/garbled_circuit_util.h>
#include <garbled_circuit/garbled_circuit.h>
#include <util/util.h>
#include <tcpip/tcpip.h>

namespace GarbledInference::Garbling {

    enum class ROLE {BOB, ALICE};


    class TinyGarbleWrapper {
    public:
        //TODO: lots of comments

        /**
         * Access to the singleton Wrapper.
         */
        static TinyGarbleWrapper &getInstance();

        TinyGarbleWrapper() = delete;

        TinyGarbleWrapper(const TinyGarbleWrapper&) = delete;

        TinyGarbleWrapper(TinyGarbleWrapper&&) = delete;

        template<ROLE role>
        std::string evaluate(const std::string &input);

        TinyGarbleWrapper& operator=(const TinyGarbleWrapper&) = delete;

        TinyGarbleWrapper& operator=(TinyGarbleWrapper&&) = delete;

    private:
        int _port;
        std::string _scd_file_address;
        std::string _server_ip;
        std::string _p_init_str;
        std::string _p_input_str;
        std::string _init_str;
        //std::string _input_f_hex_str; is passed in evaluate()
        int64_t _terminate_period;
        uint64_t _clock_cycles;
        std::string _output_mask;
        OutputMode _output_mode;
        bool _disable_OT = false;
        bool _low_mem_foot = false;

        explicit TinyGarbleWrapper(std::string serverAddr, const int& serverPort, std::string scdFileLocation);
    };
/*
    template<>
    std::string TinyGarbleWrapper::evaluate<ROLE::ALICE>(const std::string &input);
    template<>
    std::string TinyGarbleWrapper::evaluate<ROLE::BOB>(const std::string &input);
    */
}
