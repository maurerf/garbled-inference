#include "Client.h"

#include <utility>

#define TCP_SERVER_ADDRESS "localhost"
#define TCP_SERVER_PORT "1337"

//#define DEBUG_TCP_CLIENT

namespace GarbledInference::Networking {

    Client::Client(MessageHandler messageHandler) : _ioContext(), _socket(_ioContext), _messageHandler(std::move(messageHandler)) {
        boost::asio::ip::tcp::resolver resolver { _ioContext };
        _endpoints = resolver.resolve(TCP_SERVER_ADDRESS, TCP_SERVER_PORT);
    }

    void Client::start() {
        boost::asio::async_connect(_socket, _endpoints, [this](boost::system::error_code ec, boost::asio::ip::tcp::endpoint) {
            if (!ec)
                read();
        });

        _ioContext.run();
    }

    void Client::stop() {
        boost::system::error_code ec;
        _socket.close(ec);
    }

    void Client::post(const std::string &message) {
        _outgoingMessage.emplace(message);

        write();
    }

    void Client::read() {
        boost::asio::async_read_until(_socket, _streamBuf, "\n", [this](boost::system::error_code ec, size_t /*bytesTransferred*/) {
            // on read hook
            if (ec) {
                stop();
                return;
            }

            std::stringstream message;
            message << std::istream{&_streamBuf}.rdbuf();
            _messageHandler(message.str());
            read();
        });
    }


    //TODO: make write blocking on client side
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
