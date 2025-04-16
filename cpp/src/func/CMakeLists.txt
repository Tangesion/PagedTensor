cmake_minimum_required(VERSION 3.5)
project(func)
set(SRC
    func.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fopenmp")

add_library(func SHARED ${SRC})

target_include_directories(func PUBLIC ${HEAD_PATH})


target_link_libraries(func PUBLIC runtime)

set_target_properties(func PROPERTIES POSITION_INDEPENDENT_CODE ON)