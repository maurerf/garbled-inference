#pragma once

#include <iostream>
#include <boost/asio.hpp>

#define DEBUG_TCP_SERVER

namespace GarbledInference {
    //TODO: move a lot of stuff to GarbledInference::NeuralNetworking
    namespace Networking {
        class Server {
        public:
            /**
             * Basic default constructor to init TCP server accepting incoming connections.
             */
            Server();

            Server(const Server&) = delete;

            Server(Server&&) = delete;

            Server& operator=(const Server&) = delete;

            Server& operator=(Server&&) = delete;
        };
    }
}
