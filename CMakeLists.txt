cmake_minimum_required(VERSION 3.4...3.18)
project(ftorch)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(pybind11)
pybind11_add_module(ftorch src/main.cpp src/tensor.h src/tensor.cpp src/ops/ops.h src/ops/ops.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ftorch
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
