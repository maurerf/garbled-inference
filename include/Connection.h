#pragma once

#include <boost/asio.hpp>
#include <memory>
#include <optional>
#include <iostream>

namespace GarbledInference::Networking {
    typedef std::function<void(std::string)> MessageHandler;
    typedef std::function<void()> ErrorHandler;

    class Connection : public std::enable_shared_from_this<Connection> {
    public:

        static std::shared_ptr<Connection> Create(boost::asio::ip::tcp::socket&& socket) {
            return std::shared_ptr<Connection> {new Connection(std::move(socket)) };
        }

        [[nodiscard]] std::string getIdentifier() const {
            return _identifier;
        }

        void start(MessageHandler&& messageHandler, ErrorHandler&& errorHandler);
        void post(const std::string& message);

    private:
        explicit Connection(boost::asio::ip::tcp::socket&& socket);

        // Wait for a new message from client
        void asyncRead();

        void asyncWrite();

        boost::asio::ip::tcp::socket _socket;
        std::string _identifier;

        std::optional<std::string> _outgoingMessage;
        boost::asio::streambuf _streamBuf {65536};

        MessageHandler _messageHandler;
        ErrorHandler _errorHandler;
    };
}
