cmake_minimum_required(VERSION 3.0)
project(kernel_launch)

set(SRC
    attention.cpp
    ffn.cpp
    matmul.cpp
    rmsnorm.cpp
    rope.cpp
    transpose.cpp
)

set(LIBRARY_OUTPUT_PATH ${LIB_PATH})

add_library(kernel_launch SHARED ${SRC})

target_include_directories(kernel_launch PUBLIC ${HEAD_PATH})

# 链接 runtime 和 kernel_cpu 库
target_link_libraries(kernel_launch runtime kernel_cpu func)

set_target_properties(kernel_launch PROPERTIES POSITION_INDEPENDENT_CODE ON)