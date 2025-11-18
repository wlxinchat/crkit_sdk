cmake_minimum_required(VERSION 3.16)

# 查找依赖
find_package(OpenCV QUIET)

# 基础推理示例
add_executable(basic_inference basic_inference.cpp)
target_link_libraries(basic_inference crkit_core)

if(TARGET crkit_tensorrt)
    target_link_libraries(basic_inference crkit_tensorrt)
endif()

if(TARGET crkit_onnxruntime)
    target_link_libraries(basic_inference crkit_onnxruntime)
endif()

if(OpenCV_FOUND)
    target_link_libraries(basic_inference ${OpenCV_LIBS})
endif()

# 高级Pipeline示例
add_executable(advanced_pipeline advanced_pipeline.cpp)
target_link_libraries(advanced_pipeline crkit_core)

if(TARGET crkit_tensorrt)
    target_link_libraries(advanced_pipeline crkit_tensorrt)
endif()

if(TARGET crkit_onnxruntime)
    target_link_libraries(advanced_pipeline crkit_onnxruntime)
endif()

if(OpenCV_FOUND)
    target_link_libraries(advanced_pipeline ${OpenCV_LIBS})
endif()

# 批量推理示例
add_executable(batch_inference batch_inference.cpp)
target_link_libraries(batch_inference crkit_core)
target_link_libraries(batch_inference stdc++fs)  # C++17 filesystem

if(TARGET crkit_tensorrt)
    target_link_libraries(batch_inference crkit_tensorrt)
endif()

if(TARGET crkit_onnxruntime)
    target_link_libraries(batch_inference crkit_onnxruntime)
endif()

if(OpenCV_FOUND)
    target_link_libraries(batch_inference ${OpenCV_LIBS})
endif()

# 目标检测示例
add_executable(object_detection_example object_detection_example.cpp)
target_link_libraries(object_detection_example crkit_core)

if(TARGET crkit_tensorrt)
    target_link_libraries(object_detection_example crkit_tensorrt)
endif()

if(TARGET crkit_onnxruntime)
    target_link_libraries(object_detection_example crkit_onnxruntime)
endif()

if(OpenCV_FOUND)
    target_link_libraries(object_detection_example ${OpenCV_LIBS})
endif()

# 安装示例
install(TARGETS basic_inference advanced_pipeline batch_inference object_detection_example
    RUNTIME DESTINATION bin/examples
)
