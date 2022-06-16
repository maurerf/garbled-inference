#include "Server.h"

GarbledInference::Networking::Server::Server() {
    using boost::asio::ip::tcp;

    try {
        // init tcp objects
        boost::asio::io_context ioContext;
        //TODO: ipv6?
        tcp::acceptor acceptor { ioContext, tcp::endpoint(tcp::v4(), 1337) };

#ifdef  DEBUG_TCP_SERVER
        std::cout << "Init server!" << std::endl;
#endif

        while(true) {
            tcp::socket sock { ioContext };
            acceptor.accept(sock);

#ifdef  DEBUG_TCP_SERVER
            std::cout << "Client connected!" << std::endl;
#endif

            const std::string test_msg { "Hello Client!" };
            boost::system::error_code error;

            boost::asio::write(sock, boost::asio::buffer(test_msg), error);


        }

    } catch (std::exception& e) {
        //TODO: search for other places in this project where cerr should be used instead of cout
        std::cerr << e.what() << std::endl;
    }

}
