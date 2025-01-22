find_package(pybind11 REQUIRED)

set(SRC extension.cpp)
set(TORCH_INSTALL_PREFIX "/home/tgx/Downloads/anaconda/yes/envs/cuda/lib/python3.10/site-packages/torch")
set(Torch_DIR "/home/tgx/Downloads/anaconda/yes/envs/cuda/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "TORCH_INSTALL: ${TORCH_INSTALL_PREFIX}")

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Could not find torch_python library")
endif()
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${HEAD_PATH})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})


pybind11_add_module(attentionClass ${SRC})

target_link_directories(attentionClass PUBLIC ${LIB_PATH})

target_link_libraries(attentionClass PUBLIC model ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set_target_properties(attentionClass PROPERTIES POSITION_INDEPENDENT_CODE ON)
