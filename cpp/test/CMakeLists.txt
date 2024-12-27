# 收集测试源文件
#aux_source_directory(. TEST_SOURCES)
set(TEST_SOURCES
    test.cpp
)

set(CMAKE_PREFIX_PATH "/home/gexingt/tgx/lib")

add_subdirectory(${CPP_PATH}/thirdparty/googletest-1.15.2 ${CMAKE_BINARY_DIR}/googletest)

find_package(nlohmann_json 3.2.0 REQUIRED)

# 设置编译类型为 Debug
#set(CMAKE_BUILD_TYPE Debug)

# 添加调试信息并禁用优化
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

find_package(OpenMP REQUIRED)

# 设置编译器标志以启用 OpenMP
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# 创建一个可执行文件
add_executable(test_suite ${TEST_SOURCES})


# 包含头文件目录（局部作用范围）
target_include_directories(test_suite PRIVATE ${HEAD_PATH})
target_include_directories(test_suite PRIVATE ${CPP_PATH}/thirdparty/googletest-1.15.2/googletest/include)

# 链接 runtime 库
target_link_libraries(test_suite PRIVATE runtime nlohmann_json::nlohmann_json gtest gtest_main OpenMP::OpenMP_CXX)

add_test(NAME test_suite COMMAND test_suite)