cmake_minimum_required(VERSION 3.16...3.27)
project(crkit_sdk VERSION 1.0.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译器版本要求
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.5")
        message(FATAL_ERROR "GCC >= 7.5 required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
        message(FATAL_ERROR "Clang >= 6.0 required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 选项
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_OPENVINO "Enable OpenVINO backend" OFF)

# 查找依赖（精确版本）
find_package(OpenCV 4.2 REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)

# 包含路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
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
    src/postprocess/detection_postprocessor.cpp
)

# 核心库
add_library(crkit_core ${CORE_SOURCES})
target_link_libraries(crkit_core
    ${OpenCV_LIBS}
    Threads::Threads
)

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
    # 最低版本要求
    set(ONNXRUNTIME_MIN_VERSION "1.10.0")

    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
    )
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS
            /usr/local/include
            /usr/include
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
    )

    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
        message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")
        message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime minimum required version: ${ONNXRUNTIME_MIN_VERSION}")

        add_library(crkit_onnxruntime
            src/engines/onnxruntime/onnx_engine.cpp
        )
        target_include_directories(crkit_onnxruntime PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(crkit_onnxruntime crkit_core ${ONNXRUNTIME_LIB})
        target_compile_definitions(crkit_core PUBLIC ENABLE_ONNXRUNTIME)
    else()
        message(WARNING "ONNX Runtime not found, disabling ONNX backend")
        message(WARNING "Please install ONNX Runtime >= ${ONNXRUNTIME_MIN_VERSION}")
        set(ENABLE_ONNXRUNTIME OFF)
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
