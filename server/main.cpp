// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.

#include <iostream>
#include <fstream>
#include <array>
#include <boost/asio.hpp>
#include <TinyGarbleWrapper.h>
#include <Server.h>

#include "NeuralNetConfig.h"
#include "NeuralNet.h"


/**
 * Simple I/O utility converting the MNIST image and label input files to Eigen3 matrix.
 *
 * // cf. https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
 *
 * @param inputImages relative path to input .idx3-ubyte image file
 * @param inputLabels relative path to input .idx1-ubyte label file
 * @return list of image/label pairs
 */
std::vector<std::pair<GarbledInference::Neurons, Eigen::Index>> read_mnist(const char* inputImages, const char* inputLabels);

/*
 * Main function of Garbled Inference server application.
 */
int main() {

    // read test data and labels from input file
    auto inputList= read_mnist(
            "/home/fdm/Documents/BA/git/garbled-inference/models/t10k-images.idx3-ubyte",
            "/home/fdm/Documents/BA/git/garbled-inference/models/t10k-labels.idx1-ubyte"
    );

    // check if MNIST input was properly read
    if(inputList.empty()) {
        std::cout << "Error: Input data or label file was not found. Make sure to adjust location in garbled-inference/server/main.cpp." << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t imagesClassified = 0;
    size_t imagesCorrectlyClassified = 0;
    std::chrono::steady_clock::time_point timeBegin = std::chrono::steady_clock::now();
    for(const auto &input : inputList) {
        const auto result = GarbledInference::NeuralNet::getInstance().inference(input.first);
        imagesClassified++;
        if(result == input.second) {
            imagesCorrectlyClassified++;
        }
        std::chrono::steady_clock::time_point timeIteration = std::chrono::steady_clock::now();
        std::cout
            << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeIteration - timeBegin).count() << "[ms]."
            << "Images classified: " << imagesClassified << ". "
            << "Classification Result: " << result << " : " << input.second << " : " << static_cast<float>(imagesCorrectlyClassified)/static_cast<float>(imagesClassified) << std::endl;
    }

    /*
        using namespace GarbledInference::Networking;

        [[maybe_unused]] Server server
        {
            [](std::shared_ptr<Connection> connection) {
                std::cout << "Server: New connection opened: " << connection->getIdentifier() << std::endl;
            },
            [](std::shared_ptr<Connection> connection) {
                std::cout << "Server: Connection terminated: " << connection->getIdentifier() << std::endl;
            },
            [](const std::string& message) {
                std::cout << "Server: New message: " << message << std::endl;
            },
            boost::asio::ip::tcp::v4(),
            1337
        };

        server.run();
    */

    return 0;
}



std::vector<std::pair<GarbledInference::Neurons, Eigen::Index>> read_mnist(const char *const inputImages, const char* const inputLabels)
{
    const auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::vector<std::pair<GarbledInference::Neurons, Eigen::Index>> ans;

    std::ifstream imageFile (inputImages, std::ios::binary);
    std::ifstream labelsFile (inputLabels, std::ios::binary);
    if (imageFile.is_open() and labelsFile.is_open())
    {
        //read image file header
        int magic_number_images=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        imageFile.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
        magic_number_images= reverseInt(magic_number_images);
        imageFile.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        imageFile.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        imageFile.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        //read label file header
        int magic_number_labels=0;
        int number_of_labels=0;
        labelsFile.read(reinterpret_cast<char*>(&magic_number_labels), 4);
        labelsFile.read(reinterpret_cast<char*>(&number_of_labels), 4);
        number_of_labels = reverseInt(number_of_labels);
        /*
        std::cout << "magic_number_images: " << magic_number_images << std::endl;
        std::cout << "number_of_images: " << number_of_images << std::endl;
        std::cout << "n_rows: " << n_rows << std::endl;
        std::cout << "n_cols: " << n_cols << std::endl;
         */

        for(int i=0;i<number_of_images;++i)
        {
            GarbledInference::Neurons image = {Eigen::Matrix<double, 28,28>::Zero()}; //must be hardcoded 28 because n_rows & n_cols is not defined at runtime
            //read image
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    imageFile.read((char*)&temp, sizeof(temp));
                    image.front()(r,c) = static_cast<double>(temp)/255;
                    //std::cout << temp << " ";
                }
                //std::cout << std::endl;
            }

            //read corresponding label
            unsigned char label=0;
            labelsFile.read((char*)&label, 1);
            //std::cout << static_cast<int>(label) << std::endl << std::endl;
            ans.emplace_back(std::make_pair(image, static_cast<Eigen::Index>(label)));
        }
    }
    return ans;
}

