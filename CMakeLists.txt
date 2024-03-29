# This file specifies how the project should be built, using CMake.
# If you are unfamiliar with CMake, don't worry about all the details.
# The sections you might want to edit are marked as such, and
# the comments should hopefully make most of it clear.
#
# For many purposes, you may not need to change anything about this file.

cmake_minimum_required(VERSION 3.14)

# Set project name, version and laguages here. (change as needed)
# Version numbers are available by including "exampleConfig.h" in
# the source. See exampleConfig.h.in for some more details.
project(GARBLED_INFERENCE VERSION 0.1 LANGUAGES CXX)


# Options: Things you can set via commandline options to cmake (e.g. -DENABLE_LTO=[ON|OFF])
option(ENABLE_WARNINGS_SETTINGS "Allow target_set_warnings to add flags and defines.
                                 Set this to OFF if you want to provide your own warning parameters." ON)
option(ENABLE_LTO "Enable link time optimization" ON)

# Include stuff. No change needed.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(ConfigSafeGuards)
include(Colors)
include(Documentation)
include(LTO)
include(Misc)
include(Warnings)

# Add TinyGarble subdir
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TinyGarble/)

# Check for LTO support.
find_lto(CXX)

# --------------------------------------------------------------------------------
#                         Locate files (change as needed).
# --------------------------------------------------------------------------------
set(SOURCES          # All .cpp files in src/
    src/NeuralNet.cpp
        include/Layer.h src/Layer.cpp src/Server.cpp include/Server.h src/Connection.cpp include/Connection.h include/Client.h src/Client.cpp include/weights/MNIST/convolution_2.h.in include/weights/MNIST/dense_1.h.in include/weights/MNIST/addition_1.h.in include/weights/MNIST/addition_2.h.in include/weights/MNIST/addition_3.h.in include/weights/MNIST/reshape_1.h.in include/TinyGarbleWrapper.h src/TinyGarbleWrapper.cpp)
set(LIBRARY_NAME gi-lib)  # Default name for the library built from src/*.cpp (change if you wish)


# --------------------------------------------------------------------------------
#                            Build! (Change as needed)
# --------------------------------------------------------------------------------
# Compile all sources into a library.
add_library(${LIBRARY_NAME} OBJECT ${SOURCES})

# Lib needs its header files, and users of the library must also see these (PUBLIC). (No change needed)
target_include_directories(${LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/TinyGarble)
target_link_libraries(${LIBRARY_NAME} TinyGarble)


# Set the compile options you want (change as needed).
target_set_warnings(${LIBRARY_NAME} ENABLE ALL AS_ERROR ALL DISABLE Annoying)
# target_compile_options(${LIBRARY_NAME} ... )  # For setting manually.

# Add an executable for the file app/main.cpp.
# If you add more executables, copy these lines accordingly.
add_executable(client app/main.cpp)   # Name of exec. and location of file.
target_link_libraries(client PRIVATE ${LIBRARY_NAME})  # Link the executable to library (if it uses it).
target_set_warnings(client ENABLE ALL AS_ERROR ALL DISABLE Annoying) # Set warnings (if needed).
target_enable_lto(client optimized)  # enable link-time-optimization if available for non-debug configurations

add_executable(server server/main.cpp)   # Name of exec. and location of file.
target_link_libraries(server PRIVATE ${LIBRARY_NAME})  # Link the executable to library (if it uses it).
target_set_warnings(server ENABLE ALL AS_ERROR ALL DISABLE Annoying) # Set warnings (if needed).
target_enable_lto(server optimized)  # enable link-time-optimization if available for non-debug configurations

# Set the properties you require, e.g. what C++ standard to use. Here applied to library and main (change as needed).
set_target_properties(
    ${LIBRARY_NAME} client
      PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

set_target_properties(
        ${LIBRARY_NAME} server
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
