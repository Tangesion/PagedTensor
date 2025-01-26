# 收集测试源文件
#aux_source_directory(. TEST_SOURCES)


set(CMAKE_PREFIX_PATH "/home/gexingt/tgx/lib")

add_subdirectory(${CPP_PATH}/thirdparty/googletest-1.15.2 ${CMAKE_BINARY_DIR}/googletest)

#find_package(nlohmann_json 3.2.0 REQUIRED)

# 设置编译类型为 Debug
#set(CMAKE_BUILD_TYPE Debug)

# 添加调试信息并禁用优化
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

#find_package(OpenMP REQUIRED)

# 设置编译器标志以启用 OpenMP


# 定义测试源文件列表
set(TEST_SOURCES
    attention.cpp
    matmul.cpp
    rmsnorm.cpp
    rope.cpp
    ffn.cpp
    transpose.cpp
    tensorTest.cpp
)


# 为每个测试源文件创建一个可执行文件
foreach(TEST_SOURCE ${TEST_SOURCES})
    # 获取文件名（不带扩展名）
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # 创建可执行文件
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # 包含头文件目录（局部作用范围）
    target_include_directories(${TEST_NAME} PRIVATE ${HEAD_PATH})
    target_include_directories(${TEST_NAME} PRIVATE ${CPP_PATH}/thirdparty/googletest-1.15.2/googletest/include)
    
    # 链接库
    target_link_libraries(${TEST_NAME} PRIVATE func kernel_launch gtest gtest_main)
    
    # 添加测试
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()