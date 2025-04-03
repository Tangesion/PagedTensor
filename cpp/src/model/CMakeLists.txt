cmake_minimum_required(VERSION 3.5)
project(model)

set(SRC
    llama2.cpp
)

set(LIBRARY_OUTPUT_PATH ${LIB_PATH})

add_library(model SHARED ${SRC})

target_include_directories(model PUBLIC ${HEAD_PATH})

# 链接 runtime 和 kernel_launch 库
target_link_libraries(model runtime kernel_launch func)

set_target_properties(model PROPERTIES POSITION_INDEPENDENT_CODE ON)