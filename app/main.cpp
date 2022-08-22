// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef /*ENABLE_DOCTEST_IN_LIBRARY*/ false //TODO: fix doctest
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif


#include <TinyGarbleWrapper.h>
#include <iostream>
#include <chrono> // std::chrono::microseconds
#include <thread> // std::this_thread::sleep_for
/*
 * Main function of Garbled Inference user application.
 */
int main() {
    while(true) {
        std::cout << GarbledInference::Garbling::TinyGarbleWrapper::getInstance().evaluate<GarbledInference::Garbling::ROLE::ALICE>("DEADBEEFDEADBEEF") << std::endl << std::endl;
        //using namespace std::chrono_literals;
        //std::this_thread::sleep_for(200ms);
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
