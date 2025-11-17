# CRKIT SDK 快速开始指南

## 目录
- [系统要求](#系统要求)
- [安装依赖](#安装依赖)
- [编译SDK](#编译sdk)
- [YOLO11快速开始](#yolo11快速开始)
- [常见问题](#常见问题)

---

## 系统要求

### 操作系统
- Ubuntu 18.04+ / CentOS 7+
- Linux Kernel 4.4+

### 编译工具
- GCC 7.5+ 或 Clang 6.0+
- CMake 3.16+
- Make 或 Ninja

### 必须依赖
- **OpenCV 4.0+**: 图像处理
- **ONNX Runtime 1.10+**: 推理引擎

### 可选依赖
- CUDA 11.0+ (GPU加速)
- TensorRT 8.0+ (NVIDIA GPU高性能推理)
- OpenVINO 2022+ (Intel平台优化)

---

## 安装依赖

### Ubuntu/Debian

```bash
# 更新软件包
sudo apt update

# 安装编译工具
sudo apt install -y build-essential cmake git

# 安装OpenCV
sudo apt install -y libopencv-dev

# 安装ONNX Runtime
# 方法1: 从官网下载预编译包
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/
sudo ldconfig

# 方法2: 使用包管理器（如果可用）
# sudo apt install -y libonnxruntime-dev
```

### CentOS/RHEL

```bash
# 安装EPEL仓库
sudo yum install -y epel-release

# 安装编译工具
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3

# 安装OpenCV
sudo yum install -y opencv-devel

# 安装ONNX Runtime (需要手动下载安装，参考Ubuntu步骤)
```

---

## 编译SDK

### 1. 克隆仓库

```bash
git clone https://github.com/yourorg/crkit_sdk.git
cd crkit_sdk
```

### 2. 创建构建目录

```bash
mkdir build
cd build
```

### 3. 配置CMake

```bash
# 基础配置（仅CPU，ONNX Runtime）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DENABLE_ONNXRUNTIME=ON \
  -DENABLE_TENSORRT=OFF

# GPU加速配置（如果有NVIDIA GPU和CUDA）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DENABLE_ONNXRUNTIME=ON \
  -DENABLE_TENSORRT=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### 4. 编译

```bash
make -j$(nproc)
```

### 5. 安装（可选）

```bash
sudo make install
```

---

## YOLO11快速开始

### 1. 准备模型

#### 导出YOLO11模型为ONNX格式

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo11n.pt')  # 或你自己训练的模型

# 导出为ONNX格式
model.export(format='onnx',
             imgsz=640,
             simplify=True,
             opset=12)

# 导出成功后会生成 yolo11n.onnx 文件
```

### 2. 准备测试图像

准备一张待检测的图像，例如 `test_image.jpg`。

### 3. 运行YOLO11检测示例

```bash
# 在build目录下运行
./examples/yolo11_detection ../models/yolo11n.onnx test_image.jpg 0.5 0.45

# 参数说明:
# - ../models/yolo11n.onnx: ONNX模型路径
# - test_image.jpg: 输入图像路径
# - 0.5: 置信度阈值
# - 0.45: NMS IOU阈值
```

### 4. 查看结果

程序会输出检测结果并保存可视化图像：

```
=== Detection Results ===
Total detections: 3
Inference time: 25.3 ms
FPS: 39.5

Detected defects:
--------------------------------------------------------------------------------
   ID               Class  Confidence              X         Y     Width    Height
--------------------------------------------------------------------------------
    1             scratch       0.852          125.3      89.2     156.7     203.4
    2                dent       0.734          450.1     320.5      89.3     112.8
    3               crack       0.681          200.5     180.2     234.1     156.9
--------------------------------------------------------------------------------

Result saved to: detection_result.jpg
```

---

## 使用SDK进行开发

### 基础示例

```cpp
#include <crkit/crkit.h>
#include <crkit/postprocess/yolo_postprocessor.h>

int main() {
    // 1. 初始化SDK
    crkit::Initialize();

    // 2. 配置YOLO后处理
    crkit::postprocess::YOLOConfig yolo_config;
    yolo_config.num_classes = 80;
    yolo_config.conf_threshold = 0.5f;
    yolo_config.iou_threshold = 0.45f;

    auto yolo_postprocessor =
        std::make_shared<crkit::postprocess::YOLOPostProcessor>(yolo_config);

    // 3. 创建推理引擎
    crkit::InferenceConfig config;
    config.engine_type = crkit::EngineType::ONNXRUNTIME;

    auto engine = crkit::CreateInferenceEngine(
        config.engine_type,
        "yolo11n.onnx",
        config
    );

    // 4. 加载图像
    crkit::utils::ImageLoadOptions load_options;
    load_options.target_width = 640;
    load_options.target_height = 640;
    load_options.hwc_to_chw = true;

    crkit::Tensor input = crkit::utils::LoadImage("image.jpg", load_options);

    // 添加batch维度
    crkit::Tensor batch_input({1, input.Dim(0), input.Dim(1), input.Dim(2)},
                               crkit::DataType::FLOAT32);
    std::memcpy(batch_input.Data(), input.Data(), input.ByteSize());

    // 5. 推理
    crkit::InferenceResult raw_result;
    engine->Infer(batch_input, raw_result);

    // 6. 后处理
    crkit::InferenceResult final_result;
    yolo_postprocessor->Process(raw_result.raw_outputs, final_result);

    // 7. 使用结果
    for (const auto& det : final_result.detections) {
        std::cout << det.label << ": " << det.confidence << std::endl;
    }

    // 8. 清理
    crkit::Shutdown();
    return 0;
}
```

### 编译你的程序

```bash
# 使用pkg-config
g++ -std=c++17 my_app.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -lcrkit_core -lcrkit_onnxruntime \
    -lonnxruntime \
    -o my_app

# 或使用CMake
```

CMakeLists.txt示例：

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(crkit REQUIRED)  # 如果SDK已安装

add_executable(my_app my_app.cpp)
target_link_libraries(my_app
    crkit::crkit_core
    crkit::crkit_onnxruntime
    ${OpenCV_LIBS}
)
```

---

## 工业质检应用示例

### 批量质检流程

```cpp
// 配置质检参数
crkit::postprocess::YOLOConfig config;
config.num_classes = 8;  // 8种缺陷类型
config.conf_threshold = 0.3f;  // 质检通常使用较低阈值
config.class_names = {
    "scratch", "dent", "crack", "discoloration",
    "contamination", "burr", "bubble", "deformation"
};

// 加载模型
auto engine = crkit::CreateInferenceEngine(
    crkit::EngineType::ONNXRUNTIME,
    "defect_detector.onnx"
);

// 批量检测
std::vector<std::string> image_files = GetImageFiles("./images/");

for (const auto& image_file : image_files) {
    // 加载图像
    auto input = LoadAndPreprocess(image_file);

    // 推理
    crkit::InferenceResult result;
    engine->Infer(input, result);

    // 判断是否合格
    if (result.detections.empty()) {
        std::cout << image_file << ": PASS" << std::endl;
    } else {
        std::cout << image_file << ": FAIL - "
                  << result.detections.size() << " defects" << std::endl;

        // 记录缺陷信息
        for (const auto& det : result.detections) {
            LogDefect(image_file, det.label, det.confidence);
        }
    }
}
```

---

## 常见问题

### Q1: 找不到ONNX Runtime库

**错误信息:**
```
CMake Error: Could not find ONNX Runtime
```

**解决方法:**
```bash
# 确保ONNX Runtime已正确安装
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 或在CMake中指定路径
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

### Q2: 推理速度慢

**解决方法:**
1. 确保使用Release模式编译：`-DCMAKE_BUILD_TYPE=Release`
2. 增加线程数：`config.num_threads = 8;`
3. 使用GPU加速（如果有NVIDIA GPU）
4. 使用TensorRT引擎（性能最佳）

### Q3: 检测结果不准确

**解决方法:**
1. 调整置信度阈值：`config.conf_threshold = 0.3f;`
2. 调整NMS阈值：`config.iou_threshold = 0.5f;`
3. 确保输入图像预处理正确（尺寸、归一化等）
4. 检查类别名称是否匹配训练时的类别

### Q4: 内存占用过高

**解决方法:**
1. 限制同时加载的模型数量：
   ```cpp
   ModelManager::Instance().SetMaxLoadedModels(3);
   ```
2. 及时卸载不用的模型：
   ```cpp
   ModelManager::Instance().UnloadModel("model_id");
   ```
3. 定期清理idle模型：
   ```cpp
   ModelManager::Instance().CleanupUnusedModels(300); // 5分钟
   ```

### Q5: 如何使用自己训练的YOLO11模型？

**步骤:**
1. 确保模型输出格式兼容（通常YOLO11输出格式为 `[batch, 84, 8400]` 对于COCO数据集）
2. 修改`num_classes`和`class_names`
3. 导出时确保使用相同的输入尺寸（640x640）

```python
# 训练自己的模型
model = YOLO('yolo11n.yaml')
model.train(data='defects.yaml', epochs=100)

# 导出为ONNX
model.export(format='onnx', imgsz=640)
```

---

## 下一步

- 查看 [API文档](API.md) 了解详细的接口说明
- 查看 [设计文档](DESIGN.md) 了解架构设计
- 查看 [examples](../examples/) 目录获取更多示例代码
- 查看 [DEPLOYMENT.md](DEPLOYMENT.md) 了解生产环境部署

---

## 获取帮助

- Email: wlxinchat@gmail.com
- 文档: 详见 docs/ 目录
- GitHub Issues: (待配置)
