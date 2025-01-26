cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(InferenceFrame VERSION 1.0)

# 开启测试
enable_testing()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(LIBRARY_OUTPUT_PATH ${LIB_PATH})

find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)

# 设置编译类型为 Debug
#set(CMAKE_BUILD_TYPE Debug)

# 添加调试信息并禁用优化
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
# 启用优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(test)