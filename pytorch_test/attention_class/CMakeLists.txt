find_package(pybind11 REQUIRED)
message(STATUS "SRC: ${CPP_PATH}/src/model/llama2.cpp")
set(SRC extension.cpp ${CPP_PATH}/src/model/llama2.cpp)

if(DEFINED ENV{TORCH_PATH})
    set(TORCH_INSTALL_PREFIX $ENV{TORCH_PATH})
    set(Torch_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
else()
    message(FATAL_ERROR "Environment variable TORCH_PATH is not set. Please set it to the path of torch.")
endif()
#set(TORCH_INSTALL_PREFIX "/home/tgx/Downloads/anaconda/yes/envs/cuda/lib/python3.10/site-packages/torch")
#set(Torch_DIR "/home/tgx/Downloads/anaconda/yes/envs/cuda/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "TORCH_INSTALL: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch_DIR: ${Torch_DIR}")

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Could not find torch_python library")
endif()
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${HEAD_PATH} $(CMAKE_SOURCE_DIR)/utils)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})


pybind11_add_module(attention_class ${SRC})

target_link_directories(attention_class PUBLIC ${LIB_PATH})

target_link_libraries(attention_class PUBLIC runtime kernel_launch func ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set_target_properties(attention_class PROPERTIES POSITION_INDEPENDENT_CODE ON)
