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
    }

    void Connection::post(const std::string &message) {
        _outgoingMessage.emplace(message);

        std::cout << "ich schicke nun: " << _outgoingMessage.value() << std::endl;

        asyncWrite();
    }

    void Connection::asyncRead() {
        boost::asio::async_read_until(_socket, _streamBuf, "\n", [shared_this = shared_from_this()]
                (boost::system::error_code ec, size_t bytesTransferred) {
            if (ec) {
                shared_this->_socket.close(ec);

                shared_this->_errorHandler();
                return;
            }

            std::stringstream message;
            message << shared_this->_identifier << ": " << std::istream(&(shared_this->_streamBuf)).rdbuf();
            shared_this->_streamBuf.consume(bytesTransferred);

            shared_this->_messageHandler(message.str());
            shared_this->asyncRead();
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

            std::cout << "server hat geschickt: " << _outgoingMessage.value() << std::endl;
        });
    }

}
