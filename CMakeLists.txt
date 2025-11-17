cmake_minimum_required(VERSION 3.16)
project(crkit_sdk VERSION 1.0.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 选项
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_TENSORRT "Enable TensorRT backend" ON)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_OPENVINO "Enable OpenVINO backend" OFF)

# 包含路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# 源文件
set(CORE_SOURCES
    src/core/inference_engine.cpp
    src/core/model_manager.cpp
    src/core/inference_context.cpp
    src/data/tensor.cpp
    src/data/inference_result.cpp
    src/utils/logger.cpp
    src/utils/image_utils.cpp
)

# 核心库
add_library(crkit_core ${CORE_SOURCES})

# TensorRT后端
if(ENABLE_TENSORRT)
    find_package(CUDA REQUIRED)
    find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu)

    if(TENSORRT_LIB)
        add_library(crkit_tensorrt
            src/engines/tensorrt/tensorrt_engine.cpp
            src/engines/tensorrt/tensorrt_builder.cpp
        )
        target_link_libraries(crkit_tensorrt crkit_core ${TENSORRT_LIB} ${CUDA_LIBRARIES})
        target_compile_definitions(crkit_core PUBLIC ENABLE_TENSORRT)
    endif()
endif()

# ONNX Runtime后端
if(ENABLE_ONNXRUNTIME)
    find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib)

    if(ONNXRUNTIME_LIB)
        add_library(crkit_onnxruntime
            src/engines/onnxruntime/onnx_engine.cpp
        )
        target_link_libraries(crkit_onnxruntime crkit_core ${ONNXRUNTIME_LIB})
        target_compile_definitions(crkit_core PUBLIC ENABLE_ONNXRUNTIME)
    endif()
endif()

# 主库
add_library(crkit INTERFACE)
target_link_libraries(crkit INTERFACE crkit_core)

if(TARGET crkit_tensorrt)
    target_link_libraries(crkit INTERFACE crkit_tensorrt)
endif()

if(TARGET crkit_onnxruntime)
    target_link_libraries(crkit INTERFACE crkit_onnxruntime)
endif()

# 安装
install(TARGETS crkit_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/crkit DESTINATION include)

# 示例
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
