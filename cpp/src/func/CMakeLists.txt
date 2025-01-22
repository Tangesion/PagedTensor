cmake_minimum_required(VERSION 3.0)
project(func)
set(SRC
    func.cpp
)
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(func SHARED ${SRC})

target_include_directories(func PUBLIC ${HEAD_PATH})

target_link_libraries(func runtime)

set_target_properties(func PROPERTIES POSITION_INDEPENDENT_CODE ON)