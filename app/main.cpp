// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <array>
#include <boost/asio.hpp>

#include "NeuralNetConfig.h"
#include "NeuralNet.h"

#define TCP_SERVER_ADDRESS "localhost"
#define TCP_SERVER_PORT "1337"

#define DEBUG_TCP_CLIENT

/*
 * Main function of Garbled Inference user application.
 */
int main() {

    //TODO: isolate all of this into separate class
    using boost::asio::ip::tcp;
    try {
        // init tcp objects
        boost::asio::io_context ioContext;
        tcp::resolver resolver { ioContext };
        auto endpoints { resolver.resolve(TCP_SERVER_ADDRESS, TCP_SERVER_PORT) };
        tcp::socket sock { ioContext };

        // connect to tcp socket
#ifdef DEBUG_TCP_CLIENT
        std::cout << "Garbled Inference: Connecting to server..." << std::endl;
#endif

        //TODO: persistent connections
        boost::asio::connect(sock, endpoints);

#ifdef DEBUG_TCP_CLIENT
        std::cout << "Garbled Inference: Connection established to " << TCP_SERVER_ADDRESS << ":" << TCP_SERVER_PORT << "." << std::endl;
#endif

        while(true) {
            std::array<char, 128> buf {};
            boost::system::error_code error;

            const auto len = sock.read_some(boost::asio::buffer(buf), error);

            if(error == boost::asio::error::eof) {
                break;
            } else if(error) {
                throw boost::system::system_error(error);
            }

            std::cout.write(buf.data(), static_cast<long>(len));
            std::cout << std::endl;
        }

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "GarbledInference: Could not connect to server. Make sure a server instance is running at: " << TCP_SERVER_ADDRESS << ":" << TCP_SERVER_PORT << std::endl;

        exit(EXIT_FAILURE);
    }


    // init dummy image (all zeros) and infer (TODO read from file)
    const auto result = GarbledInference::NeuralNet::getInstance().inference({Eigen::Matrix<double, 28,28>::Zero()});

    for(const auto& m : result) {
        std::cout << m << std::endl;
    }

    return 0;
}
