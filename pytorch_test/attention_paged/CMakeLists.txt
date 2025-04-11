cmake_minimum_required(VERSION 3.10)

project(attention_paged)

# 查找 pybind11
find_package(pybind11 REQUIRED)

# 检查 TORCH_PATH 环境变量
if(DEFINED ENV{TORCH_PATH})
    set(TORCH_INSTALL_PREFIX $ENV{TORCH_PATH})
    set(Torch_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
else()
    message(FATAL_ERROR "Environment variable TORCH_PATH is not set. Please set it to the path of torch.")
endif()

# 查找 torch_python 库
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "TORCH_INSTALL: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch_DIR: ${Torch_DIR}")

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Could not find torch_python library")
endif()

# 查找 Torch 和 Python3
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# 包含头文件路径
include_directories(${HEAD_PATH} $(CMAKE_SOURCE_DIR)/utils)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# 添加 pybind11 模块
pybind11_add_module(attention_paged extension.cpp)
message(STATUS "LIB_PATH: ${LIB_PATH}")
target_link_directories(attention_paged PUBLIC ${LIB_PATH})

# 链接库
target_link_libraries(attention_paged PUBLIC runtime kernel_launch func ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# 设置目标属性
set_target_properties(attention_paged PROPERTIES POSITION_INDEPENDENT_CODE ON)