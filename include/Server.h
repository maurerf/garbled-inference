#pragma once

#include <iostream>
#include <boost/asio.hpp>
#include <optional>
#include <unordered_set>

#include "Connection.h"

//#define DEBUG_TCP_SERVER

namespace GarbledInference::Networking {
        class Server {

        public:
            typedef std::function<void(std::shared_ptr<Connection>)> JoinHandler;
            typedef std::function<void(std::shared_ptr<Connection>)> LeaveHandler;
            typedef std::function<void(std::string)> MessageHandler;

            /**
             * TODO
             * @param ipVersion IPV4 or IPV6
             * @param portNum IP port to be used
             */
            Server(JoinHandler joinHandler,
                   LeaveHandler leaveHandler,
                   MessageHandler messageHandler,
                   boost::asio::ip::tcp ipVersion,
                   boost::asio::ip::port_type portNum
                   );

            /*
            Server() = delete;

            Server(const Server&) = delete;

            Server(Server&&) = delete;

            Server& operator=(const Server&) = delete;

            Server& operator=(Server&&) = delete;
             */

            void run();

        private:
            boost::asio::io_context _ioContext;
            boost::asio::ip::tcp::acceptor _acceptor;
            std::optional<boost::asio::ip::tcp::socket> _socket;

            void acceptNextConnection();

            JoinHandler _joinHandler;
            LeaveHandler _leaveHandler;
            MessageHandler _messageHandler;

            std::unordered_set<std::shared_ptr<Connection>> _connections;
        };
    }
