#include "Server.h"

#include <utility>

GarbledInference::Networking::Server::Server(JoinHandler joinHandler, LeaveHandler leaveHandler, MessageHandler messageHandler, boost::asio::ip::tcp ipVersion, boost::asio::ip::port_type portNum) :
_ioContext(),
_acceptor(_ioContext, boost::asio::ip::tcp::endpoint(ipVersion, portNum)),
_joinHandler(std::move(joinHandler)),
_leaveHandler(std::move(leaveHandler)),
_messageHandler(std::move(messageHandler)),
_connections()
{ }

void GarbledInference::Networking::Server::run() {
    try {
        startAccept();
        _ioContext.run();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void GarbledInference::Networking::Server::startAccept() {
    _socket.emplace(_ioContext);
    auto connection = Connection::Create(std::move(*_socket));

    // asynchronously accept the connection
    _acceptor.async_accept(*_socket, [this, &connection](const boost::system::error_code& ec){
        // on connect hook
        if (!ec) {
            connection->start(
                    // message handler
                    [this](const std::string &message) { _messageHandler(message); },
                    // error handler
                    [&, weak = std::weak_ptr(connection)] {
                        if (auto shared = weak.lock(); shared && _connections.erase(shared)) {
                            _leaveHandler(shared);
                        }
                    }
            );
        }

        startAccept();
    });
}
