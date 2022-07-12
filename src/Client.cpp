#include "Client.h"

#include <utility>
#include <iostream>

#define TCP_SERVER_ADDRESS "localhost"
#define TCP_SERVER_PORT "1337"

//#define DEBUG_TCP_CLIENT

namespace GarbledInference::Networking {

    Client::Client(MessageHandler messageHandler) : _ioContext(), _socket(_ioContext), _messageHandler(std::move(messageHandler)) {
        boost::asio::ip::tcp::resolver resolver { _ioContext };
        _endpoints = resolver.resolve(TCP_SERVER_ADDRESS, TCP_SERVER_PORT);
    }

    void Client::start() {
        std::cout << "Client: Starting." << std::endl;
        boost::asio::async_connect(_socket, _endpoints, [](boost::system::error_code ec, boost::asio::ip::tcp::endpoint) {
            if (!ec) {
                std::cout << "Client: Connected." << std::endl;
            }
        });

        _ioContext.run();
    }

    void Client::stop() {
        boost::system::error_code ec;
        _socket.close(ec);

        std::cout << "Client: Stopping." << std::endl;
    }

    void Client::post(const std::string &message) {
        _outgoingMessage.emplace(message);
        write();
    }

    std::string Client::get() {
        read();
        //TODO: just use synchronous read?
        while(std::string_view(_incomingMessage.str()).back() != '\n') {
            std::cout << _incomingMessage.str() << std::endl;
        }
        return _incomingMessage.str();
    }

    void Client::read() {
        _incomingMessage.clear();
        boost::asio::async_read_until(_socket, _streamBuf, "\n", [this](boost::system::error_code ec, size_t /*bytesTransferred*/) {
            // on read hook
            if (ec) {
                stop();
                return;
            }
            std::cout <<  "kam was" << std::endl;

            _incomingMessage << std::istream{&_streamBuf}.rdbuf();
            _messageHandler(_incomingMessage.str());
        });
    }


    void Client::write() {
        boost::asio::async_write(_socket, boost::asio::buffer(_outgoingMessage.value()), [this](boost::system::error_code ec, size_t /*bytesTransferred*/) {
            // on write hook
            if (ec) {
                stop();
                return;
            }

            _outgoingMessage.reset();
        });
    }


}
