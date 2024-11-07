cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(InferenceFrame VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(RUNTIME_LIB runtime)

find_package(CUDA REQUIRED)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(test)