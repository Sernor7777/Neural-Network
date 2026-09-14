cmake_minimum_required(VERSION 3.20)

project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_compile_options(-O3)

add_subdirectory(neuralnetwork)
add_subdirectory(examples)