# CRKIT SDK - 工业质检AI推理SDK

## 项目简介

CRKIT SDK是一个专为工业质检场景设计的通用AI推理SDK，提供统一的接口支持多种推理引擎。

## 核心特性

- **统一接口**: 提供一致的API，支持TensorRT、ONNX Runtime、OpenVINO等多种推理引擎
- **高性能**: 零拷贝设计、内存池管理、批处理支持
- **易于集成**: 简洁的C++ API，最小化依赖
- **线程安全**: 支持多线程并发推理
- **灵活扩展**: 插件化架构，易于添加新的推理引擎
- **工业级**: 完善的错误处理、日志系统、资源管理

## 支持的推理引擎

- TensorRT (NVIDIA GPU)
- ONNX Runtime (CPU/GPU)
- OpenVINO (Intel)
- NCNN (嵌入式设备)

## 项目结构

```
crkit_sdk/
├── include/                # 公共头文件
│   ├── crkit/
│   │   ├── core/          # 核心接口
│   │   ├── data/          # 数据结构
│   │   ├── engine/        # 推理引擎接口
│   │   └── utils/         # 工具类
├── src/                   # 源代码实现
│   ├── core/
│   ├── engines/           # 各推理引擎实现
│   │   ├── tensorrt/
│   │   ├── onnxruntime/
│   │   └── openvino/
│   └── utils/
├── examples/              # 示例代码
├── tests/                 # 单元测试
├── docs/                  # 文档
└── cmake/                 # CMake配置

```

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 使用示例

```cpp
#include <crkit/inference_engine.h>

// 创建推理引擎
auto engine = crkit::CreateInferenceEngine(
    crkit::EngineType::TENSORRT,
    "model.engine"
);

// 准备输入数据
crkit::Tensor input = crkit::LoadImage("defect.jpg");

// 执行推理
auto result = engine->Infer(input);

// 处理结果
for (const auto& detection : result.detections) {
    std::cout << "Defect: " << detection.label
              << " Score: " << detection.score << std::endl;
}
```

## 系统要求

- Linux (Ubuntu 18.04+, CentOS 7+)
- GCC 7.5+ 或 Clang 6.0+
- CMake 3.16+
- CUDA 11.0+ (可选，用于GPU加速)

## License

Apache 2.0
