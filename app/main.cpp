// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif


#include <TinyGarbleWrapper.h>
#include <iostream>

/*
 * Main function of Garbled Inference user application.
 */
int main() {
    while(true) {
        std::cout << "RESULT BOB: "
                  << GarbledInference::Garbling::TinyGarbleWrapper::getInstance().evaluate<GarbledInference::Garbling::ROLE::ALICE>(
                          "010") << std::endl;
    }

    // create client & connect to server
    /*using namespace GarbledInference::Networking;
    [[maybe_unused]] Client client {
        [](const std::string& message) {
            std::cout << "Client: New message: " << message << std::endl;
        }
    };

    client.start();

    std::cout << client.get();*/

    exit(0);

}
