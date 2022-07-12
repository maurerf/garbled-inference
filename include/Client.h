#pragma once
#include <boost/asio.hpp>
#include <optional>

namespace GarbledInference::Networking {
    typedef std::function<void(std::string)> MessageHandler;

    class Client {
    public:
        //TODO: lots of comments

        explicit Client(MessageHandler messageHandler);

        void start();
        void stop();
        void post(const std::string& message);
        std::string get();

    private:
        void read();
        void write();

        boost::asio::io_context _ioContext;
        boost::asio::ip::tcp::socket _socket;
        boost::asio::ip::tcp::resolver::results_type _endpoints;

        // TODO: i forgot why this max size is necessary
        boost::asio::streambuf _streamBuf{65536};

        std::stringstream _incomingMessage {};
        std::optional<std::string> _outgoingMessage {};

        MessageHandler _messageHandler;
    };
}
