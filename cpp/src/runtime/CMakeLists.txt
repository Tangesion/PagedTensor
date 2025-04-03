cmake_minimum_required(VERSION 3.5)
project(runtime)
#aux_source_directory(. SRC)
set(SRC
    buffer.cpp
    bufferManager.cpp
    tensor.cpp
)
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(runtime SHARED ${SRC})

target_include_directories(runtime PUBLIC ${HEAD_PATH})
target_include_directories(runtime PUBLIC ${CUDA_INCLUDE_DIRS})

# 链接 CUDA 库
target_link_libraries(runtime ${CUDA_LIBRARIES})

set_target_properties(runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
