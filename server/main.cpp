// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include "Server.h"

/*
 * Main function of Garbled Inference data provider.
 */
int main() {
    using namespace GarbledInference::Networking;

    [[maybe_unused]] Server server
    {
        [](std::shared_ptr<Connection> connection) {
            std::cout << "Server: Connection started with: " << connection->getIdentifier() << std::endl;
        },
        [](std::shared_ptr<Connection> connection) {
            std::cout << "Server: Connection ended with: " << connection->getIdentifier() << std::endl;
        },
        [](const std::string& message) {
            std::cout << "Server: New message: " << message << std::endl;
        },
        boost::asio::ip::tcp::v4(),
        1337
    };

    server.run();

    return 0;
}
