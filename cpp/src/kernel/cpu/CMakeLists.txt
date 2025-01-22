cmake_minimum_required(VERSION 3.0)
project(kernel_cpu)

find_package(OpenMP REQUIRED)

set(SRC
    attention.cpp
    ffn.cpp
    matmul.cpp
    rmsnorm.cpp
    rope.cpp
    transpose.cpp
)

set(LIBRARY_OUTPUT_PATH ${LIB_PATH})

add_library(kernel_cpu SHARED ${SRC})

target_include_directories(kernel_cpu PUBLIC ${HEAD_PATH})

target_link_libraries(kernel_cpu PUBLIC OpenMP::OpenMP_CXX)

set_target_properties(kernel_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

