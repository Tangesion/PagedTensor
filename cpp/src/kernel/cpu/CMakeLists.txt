cmake_minimum_required(VERSION 3.5)
project(kernel_cpu)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

