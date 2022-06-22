#include "Connection.h"
#include <iostream>

namespace GarbledInference::Networking {
    Connection::Connection(boost::asio::ip::tcp::socket&& socket) : _socket(std::move(socket)) {

        std::stringstream name;
        name << _socket.remote_endpoint();

        _identifier = name.str();
    }

    void Connection::start(MessageHandler&& messageHandler, ErrorHandler&& errorHandler) {
        _messageHandler = std::move(messageHandler);
        _errorHandler = std::move(errorHandler);

        asyncRead();
    }

    void Connection::post(const std::string &message) {
        _outgoingMessage.emplace(message);
        asyncWrite();
    }

    void Connection::asyncRead() {
        // TODO: different delimiter?
        boost::asio::async_read_until(_socket, _streamBuf, "\n", [this]
                (boost::system::error_code ec, size_t bytesTransferred) {
            if (ec) {
                _socket.close(ec);

                _errorHandler();
                return;
            }

            std::stringstream message;
            message << _identifier << ": " << std::istream(&(_streamBuf)).rdbuf();
            _streamBuf.consume(bytesTransferred);

            _messageHandler(message.str());
            asyncRead();
        });
    }

    void Connection::asyncWrite() {
        boost::asio::async_write(_socket, boost::asio::buffer(_outgoingMessage.value()), [this]
                (boost::system::error_code ec, size_t) {
            if (ec) {
                _socket.close(ec);

                _errorHandler();
                return;
            }
        });
    }

}
