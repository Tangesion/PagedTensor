# 收集测试源文件
#aux_source_directory(. TEST_SOURCES)
set(TEST_SOURCES
    test.cpp
)
# 创建一个可执行文件
add_executable(test_suite ${TEST_SOURCES})

# 链接 runtime 库
target_link_libraries(test_suite PRIVATE runtime)

# 包含头文件目录（局部作用范围）
target_include_directories(test_suite PRIVATE ${HEAD_PATH})