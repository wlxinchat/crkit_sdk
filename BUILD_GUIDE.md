# CRKIT SDK 编译运行指南

## 快速开始

### 方式1: 简单编译测试（推荐用于快速验证）

```bash
# 1. 编译综合测试
g++ -std=c++17 -I./include -I./src \
    tests/test_comprehensive.cpp \
    src/data/tensor.cpp \
    src/data/inference_result.cpp \
    src/postprocess/detection_postprocessor.cpp \
    src/utils/logger.cpp \
    -o test_comprehensive -lpthread

# 2. 运行测试
./test_comprehensive

# 3. 编译日志系统测试
g++ -std=c++17 -I./include \
    tests/test_logger.cpp \
    -o test_logger -lpthread

# 4. 运行日志测试
./test_logger
```

### 方式2: 使用CMake完整编译（推荐用于开发）

#### 前置依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev

# 可选：安装ONNX Runtime（用于完整推理功能）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/
sudo ldconfig
```

#### 编译步骤

```bash
# 1. 创建构建目录
mkdir -p build
cd build

# 2. 配置CMake（基础版本，不需要OpenCV）
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF

# 或者配置完整版本（需要OpenCV和ONNX Runtime）
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON

# 3. 编译
make -j$(nproc)

# 4. 运行测试
ctest --output-on-failure

# 或者直接运行测试程序
./test_comprehensive
```

## 详细说明

### 编译选项

| CMake选项 | 说明 | 默认值 | 依赖 |
|-----------|------|--------|------|
| `BUILD_TESTS` | 编译测试程序 | OFF | 无 |
| `BUILD_EXAMPLES` | 编译示例程序 | ON | OpenCV, ONNX Runtime |
| `ENABLE_ONNXRUNTIME` | 启用ONNX Runtime | ON | ONNX Runtime |
| `CMAKE_BUILD_TYPE` | 构建类型 | Release | 无 |

### 测试程序说明

#### 1. test_comprehensive（综合测试）✅ 推荐

**用途**: 核心功能测试，**不需要任何外部依赖**

**测试内容**:
- Tensor基础操作和高级功能 (9项)
- BBox和Detection结构 (3项)
- IOU计算 (5项)
- NMS算法 (6项)
- InferenceResult管理 (5项)
- 性能测试 (2项)

**编译命令**:
```bash
g++ -std=c++17 -I./include -I./src \
    tests/test_comprehensive.cpp \
    src/data/tensor.cpp \
    src/data/inference_result.cpp \
    src/postprocess/detection_postprocessor.cpp \
    src/utils/logger.cpp \
    -o test_comprehensive -lpthread
```

**运行**:
```bash
./test_comprehensive
```

**预期输出**:
```
============================================
   CRKIT SDK 综合测试套件
============================================

## TC001-002: Tensor基础和高级功能测试
[测试] TC001-01: 空Tensor创建
  ✓ 通过
...

============================================
   测试统计
============================================
总测试数: 30
通过: 30 ✓
失败: 0 ✗
通过率: 100.0%
============================================

✅ 所有测试通过!
```

#### 2. test_logger（日志系统测试）✅ 推荐

**用途**: 日志系统功能测试，**不需要任何外部依赖**

**测试内容**:
- 基础日志功能（5个级别）
- 模块标识（9个模块）
- 多线程日志（3线程并发）
- 日志级别过滤
- 模块级日志控制
- 文件日志和轮转
- 彩色控制台输出
- 格式化输出

**编译命令**:
```bash
g++ -std=c++17 -I./include \
    tests/test_logger.cpp \
    -o test_logger -lpthread
```

**运行**:
```bash
./test_logger
```

**预期输出**（带颜色）:
```
========================================
  CRKIT SDK 日志系统测试
========================================

=== 测试1: 基础日志功能 ===

[2025-11-17 15:21:44.339] [INFO ] [ENGINE  ] [T:139435258292032] [onnx_engine.cpp:45] Model loaded
                              ↑        ↑            ↑                       ↑
                            颜色    模块标识      线程ID               文件:行号

...

========================================
  测试完成！
========================================
```

#### 3. test_tensor（Tensor单元测试）

**编译**:
```bash
g++ -std=c++17 -I./include \
    tests/test_tensor.cpp src/data/tensor.cpp \
    -o test_tensor -lpthread
```

#### 4. test_nms（NMS算法测试）

**编译**:
```bash
g++ -std=c++17 -I./include -I./src \
    tests/test_nms.cpp \
    src/data/tensor.cpp \
    src/data/inference_result.cpp \
    src/postprocess/detection_postprocessor.cpp \
    -o test_nms -lpthread
```

### 示例程序说明

#### object_detection_example（目标检测示例）

**依赖**: OpenCV + ONNX Runtime

**用途**: 完整的目标检测推理示例

**运行**:
```bash
# 需要准备ONNX模型文件
./object_detection_example model.onnx test_image.jpg 0.5 0.45
```

## 常见问题

### Q1: 编译时找不到OpenCV

**问题**: `Could not find a package configuration file provided by "OpenCV"`

**解决方案**:
- **方式1（推荐）**: 只编译测试，不编译示例
  ```bash
  cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF
  ```

- **方式2**: 安装OpenCV
  ```bash
  sudo apt install libopencv-dev
  ```

### Q2: 编译时找不到ONNX Runtime

**问题**: `ONNX Runtime not found`

**解决方案**:
- **方式1（推荐）**: 只编译测试，不需要ONNX Runtime
  ```bash
  # 测试程序不依赖ONNX Runtime
  g++ -std=c++17 -I./include -I./src \
      tests/test_comprehensive.cpp \
      src/data/tensor.cpp \
      src/data/inference_result.cpp \
      src/postprocess/detection_postprocessor.cpp \
      src/utils/logger.cpp \
      -o test_comprehensive -lpthread
  ```

- **方式2**: 安装ONNX Runtime（用于实际推理）
  ```bash
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
  tar -xzf onnxruntime-linux-x64-1.16.3.tgz
  sudo cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
  sudo cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/
  sudo ldconfig
  ```

### Q3: filesystem头文件找不到

**问题**: `fatal error: filesystem: No such file or directory`

**解决方案**: 需要GCC 8+或Clang 7+

```bash
# 检查GCC版本
g++ --version

# 如果版本太低，升级GCC
sudo apt install g++-9
g++-9 -std=c++17 ...
```

### Q4: 链接错误 `-lstdc++fs`

**问题**: GCC 8需要显式链接filesystem库

**解决方案**:
```bash
g++ -std=c++17 ... -lpthread -lstdc++fs
```

## 推荐的验证流程

### 步骤1: 快速验证核心功能

```bash
# 编译综合测试（无依赖）
g++ -std=c++17 -I./include -I./src \
    tests/test_comprehensive.cpp \
    src/data/tensor.cpp \
    src/data/inference_result.cpp \
    src/postprocess/detection_postprocessor.cpp \
    src/utils/logger.cpp \
    -o test_comprehensive -lpthread

# 运行测试
./test_comprehensive
```

**预期结果**: `通过率: 100.0%`

### 步骤2: 验证日志系统

```bash
# 编译日志测试（无依赖）
g++ -std=c++17 -I./include \
    tests/test_logger.cpp \
    -o test_logger -lpthread

# 运行测试
./test_logger
```

**预期结果**: 看到彩色日志输出，文件日志生成在 `logs/` 目录

### 步骤3: 验证日志文件轮转

```bash
# 检查生成的日志文件
ls -lh logs/

# 应该看到类似：
# test_logger.log.1
# test_logger.log.2
# ...
```

### 步骤4: （可选）完整编译

```bash
# 如果已安装OpenCV和ONNX Runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
make -j$(nproc)
ctest
```

## 性能基准

在标准开发机器上（Intel i7，16GB RAM）：

| 测试 | 耗时 | 说明 |
|------|------|------|
| test_comprehensive | ~50ms | 30个测试用例 |
| test_logger | ~100ms | 8个测试场景 + 文件写入 |
| NMS (1000框) | ~11ms | 性能测试 |

## 目录结构

```
crkit_sdk/
├── include/          # 头文件
├── src/             # 源文件
├── tests/           # 测试程序
├── examples/        # 示例程序（需要依赖）
├── docs/            # 文档
├── logs/            # 日志输出目录（自动创建）
└── build/           # 构建目录（需创建）
```

## 获取帮助

- 📧 Email: wlxinchat@gmail.com
- 📚 文档: docs/
- 🐛 问题: 查看各测试程序输出

## 总结

**最简单的验证方式**（零依赖）:
```bash
# 一键编译运行
g++ -std=c++17 -I./include -I./src tests/test_comprehensive.cpp src/data/tensor.cpp src/data/inference_result.cpp src/postprocess/detection_postprocessor.cpp src/utils/logger.cpp -o test_comprehensive -lpthread && ./test_comprehensive
```

**预期结果**: `✅ 所有测试通过!`
