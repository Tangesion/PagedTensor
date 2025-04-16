cmake_minimum_required(VERSION 3.10)
project(MatmulPerfTest)
set(CMAKE_PREFIX_PATH "/home/gexingt/tgx/lib")
set(TEST_SOURCES
    matmul.cpp    
    attention.cpp
    tensor.cpp
)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用调试符号以支持 perf 分析
set(CMAKE_BUILD_TYPE Debug)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fopenmp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/testperf)

foreach(TEST_SOURCE ${TEST_SOURCES})
    # 获取文件名（不带扩展名）
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    set(TEST_NAME "${TEST_NAME}_perf")
    
    # 创建可执行文件
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # 包含头文件目录（局部作用范围）
    target_include_directories(${TEST_NAME} PRIVATE ${HEAD_PATH})

    set(TEST_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME})

    file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})

    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR})

    target_link_libraries(${TEST_NAME} PRIVATE func kernel_launch)

endforeach()