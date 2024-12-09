cmake_minimum_required(VERSION 3.0)
project(runtime)
#aux_source_directory(. SRC)
set(SRC
    Buffer.cpp
    BufferManager.cpp
    Tensor.cpp
)
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(runtime SHARED ${SRC})

target_include_directories(runtime PUBLIC ${HEAD_PATH})
target_include_directories(runtime PUBLIC ${CUDA_INCLUDE_DIRS})

# 链接 CUDA 库
target_link_libraries(runtime ${CUDA_LIBRARIES})
