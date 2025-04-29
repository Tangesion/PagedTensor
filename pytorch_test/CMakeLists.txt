set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.10)

project(pytorchTest VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(LIB_PATH ${CMAKE_SOURCE_DIR}/../cpp/lib)
set(HEAD_PATH ${CMAKE_SOURCE_DIR}/../cpp/include)
set(CPP_PATH ${CMAKE_SOURCE_DIR}/../cpp)

message(STATUS "LIB_PATH: ${LIB_PATH}")
message(STATUS "HEAD_PATH: ${HEAD_PATH}")
message(STATUS "CPP_PATH: ${CPP_PATH}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fopenmp")

if(DEFINED ENV{PYBIND11_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{PYBIND11_PATH})
else()
    message(FATAL_ERROR "Environment variable PYBIND11_PATH is not set. Please set it to the path of pybind11.")
endif()

#list(APPEND CMAKE_PREFIX_PATH "/home/tgx/Downloads/anaconda/yes/envs/cuda/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

add_subdirectory(attention_class)
add_subdirectory(mlp_class)
add_subdirectory(matmul_paged)
add_subdirectory(attention_paged)
add_subdirectory(rope_paged)
add_subdirectory(attention_paged_class)