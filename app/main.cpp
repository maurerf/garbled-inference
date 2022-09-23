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
