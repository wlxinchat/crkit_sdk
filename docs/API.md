# CRKIT SDK API 参考文档

## 快速索引

- [核心类型](#核心类型)
- [数据结构](#数据结构)
- [推理引擎](#推理引擎)
- [模型管理](#模型管理)
- [处理Pipeline](#处理pipeline)
- [工具类](#工具类)

---

## 核心类型

### Status

状态类，用于错误处理。

```cpp
class Status {
public:
    Status();
    Status(StatusCode code, const std::string& msg = "");

    bool IsOK() const;
    StatusCode Code() const;
    const std::string& Message() const;

    operator bool() const;  // 隐式转换为bool
};
```

**示例:**
```cpp
Status status = engine->Infer(input, result);
if (!status.IsOK()) {
    std::cerr << "Error: " << status.Message() << std::endl;
}
```

### StatusCode

```cpp
enum class StatusCode {
    SUCCESS,
    ERROR_INVALID_PARAM,
    ERROR_MODEL_LOAD_FAILED,
    ERROR_INFERENCE_FAILED,
    ERROR_OUT_OF_MEMORY,
    ERROR_DEVICE_NOT_AVAILABLE,
    ERROR_INVALID_INPUT,
    ERROR_INVALID_OUTPUT,
    ERROR_NOT_IMPLEMENTED,
    ERROR_UNKNOWN
};
```

### EngineType

```cpp
enum class EngineType {
    TENSORRT,      // NVIDIA TensorRT
    ONNXRUNTIME,   // ONNX Runtime
    OPENVINO,      // Intel OpenVINO
    NCNN,          // Tencent NCNN
    AUTO           // 自动选择
};
```

### DeviceType

```cpp
enum class DeviceType {
    CPU,
    GPU,
    AUTO
};
```

---

## 数据结构

### Tensor

多维张量类，用于存储输入输出数据。

```cpp
class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);
    Tensor(void* data, const Shape& shape, DataType dtype, bool copy = false);

    // 数据访问
    void* Data();
    const void* Data() const;
    template<typename T> T* Data();
    template<typename T> const T* Data() const;

    // 形状信息
    const Shape& GetShape() const;
    size_t Rank() const;
    int64_t Dim(size_t idx) const;
    int64_t NumElements() const;

    // 数据类型
    DataType GetDataType() const;
    size_t ByteSize() const;

    // 操作
    bool Reshape(const Shape& new_shape);
    Tensor Clone() const;
    void Clear();
    bool Empty() const;
    template<typename T> void Fill(T value);
};
```

**示例:**
```cpp
// 创建tensor
Tensor tensor({1, 3, 640, 640}, DataType::FLOAT32);

// 访问数据
float* data = tensor.Data<float>();

// 获取形状信息
std::cout << "Shape: ";
for (auto dim : tensor.GetShape()) {
    std::cout << dim << " ";
}
```

### BBox

边界框结构。

```cpp
struct BBox {
    float x;       // 左上角x坐标
    float y;       // 左上角y坐标
    float width;   // 宽度
    float height;  // 高度

    float Area() const;
    float CenterX() const;
    float CenterY() const;
};
```

### Detection

检测结果结构。

```cpp
struct Detection {
    BBox bbox;                               // 边界框
    int class_id;                            // 类别ID
    std::string label;                       // 类别标签
    float confidence;                        // 置信度
    std::vector<KeyPoint> keypoints;         // 关键点
    std::map<std::string, float> attributes; // 其他属性
};
```

### InferenceResult

推理结果类。

```cpp
class InferenceResult {
public:
    // 结果数据
    std::vector<Detection> detections;
    std::vector<Classification> classifications;
    Segmentation segmentation;
    std::vector<Tensor> raw_outputs;

    // 统计信息
    float GetInferenceTime() const;
    void SetInferenceTime(float time_ms);

    // 操作
    void AddDetection(const Detection& det);
    void AddClassification(const Classification& cls);
    void SortDetectionsByConfidence();
    void FilterDetections(float min_confidence);
    Classification GetTopClassification() const;
    void Clear();
    bool Empty() const;
};
```

**示例:**
```cpp
InferenceResult result;
engine->Infer(input, result);

// 过滤低置信度检测
result.FilterDetections(0.5f);

// 遍历结果
for (const auto& det : result.detections) {
    std::cout << det.label << ": " << det.confidence << std::endl;
}
```

---

## 推理引擎

### InferenceConfig

推理配置结构。

```cpp
struct InferenceConfig {
    EngineType engine_type = EngineType::AUTO;
    DeviceType device_type = DeviceType::AUTO;
    int device_id = 0;
    int max_batch_size = 1;
    int num_threads = 4;
    bool enable_fp16 = false;
    bool enable_profiling = false;
    InferenceMode mode = InferenceMode::SYNC;
    size_t workspace_size = 1 << 30;
    bool use_memory_pool = true;
    std::string model_path;
    std::map<std::string, std::string> metadata;
};
```

### IInferenceEngine

推理引擎接口。

```cpp
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // 初始化
    virtual Status Initialize(const InferenceConfig& config) = 0;
    virtual Status LoadModel(const std::string& model_path) = 0;

    // 推理
    virtual Status Infer(const Tensor& input, InferenceResult& result) = 0;
    virtual Status Infer(const std::vector<Tensor>& inputs,
                        InferenceResult& result) = 0;
    virtual Status InferBatch(const std::vector<Tensor>& batch_inputs,
                             std::vector<InferenceResult>& results) = 0;
    virtual Status InferAsync(const Tensor& input,
                             InferenceCallback callback) = 0;

    // 模型信息
    virtual std::vector<std::string> GetInputNames() const = 0;
    virtual Shape GetInputShape(const std::string& name) const = 0;
    virtual DataType GetInputDataType(const std::string& name) const = 0;
    virtual std::vector<std::string> GetOutputNames() const = 0;
    virtual Shape GetOutputShape(const std::string& name) const = 0;
    virtual DataType GetOutputDataType(const std::string& name) const = 0;

    // 引擎信息
    virtual EngineType GetEngineType() const = 0;
    virtual std::string GetEngineVersion() const = 0;

    // 性能
    virtual float GetAverageInferenceTime() const = 0;
    virtual void ResetStatistics() = 0;
    virtual Status WarmUp(int iterations = 10) = 0;

    // 资源管理
    virtual void Release() = 0;
};
```

### 创建推理引擎

```cpp
// 工厂方法
std::shared_ptr<IInferenceEngine> CreateInferenceEngine(
    EngineType type,
    const std::string& model_path,
    const InferenceConfig& config = InferenceConfig()
);
```

**示例:**
```cpp
// 配置
InferenceConfig config;
config.device_type = DeviceType::GPU;
config.enable_fp16 = true;
config.max_batch_size = 8;

// 创建引擎
auto engine = CreateInferenceEngine(
    EngineType::TENSORRT,
    "model.engine",
    config
);

// 预热
engine->WarmUp(10);

// 推理
InferenceResult result;
engine->Infer(input, result);
```

---

## 模型管理

### ModelInfo

模型信息结构。

```cpp
struct ModelInfo {
    std::string model_id;
    std::string model_path;
    EngineType engine_type;
    InferenceConfig config;
    std::string description;
    std::map<std::string, std::string> metadata;
};
```

### ModelManager

模型管理器（单例）。

```cpp
class ModelManager {
public:
    static ModelManager& Instance();

    // 模型注册和加载
    Status RegisterModel(const ModelInfo& model_info);
    Status LoadModel(const std::string& model_id);
    Status UnloadModel(const std::string& model_id);

    // 获取引擎
    std::shared_ptr<IInferenceEngine> GetEngine(const std::string& model_id);
    std::shared_ptr<InferenceContext> GetContext(const std::string& model_id);

    // 查询
    bool IsModelLoaded(const std::string& model_id) const;
    ModelInfo GetModelInfo(const std::string& model_id) const;
    std::vector<std::string> ListRegisteredModels() const;
    std::vector<std::string> ListLoadedModels() const;

    // 高级功能
    Status HotSwapModel(const std::string& model_id,
                       const std::string& new_model_path);
    Status LoadModels(const std::vector<std::string>& model_ids);
    void UnloadAllModels();
    void CleanupUnusedModels(int64_t idle_time_seconds);

    // 统计
    ModelStats GetModelStats(const std::string& model_id) const;
    void SetMaxLoadedModels(size_t max_models);
};
```

**示例:**
```cpp
auto& mgr = ModelManager::Instance();

// 注册模型
ModelInfo info;
info.model_id = "detector_v1";
info.model_path = "detector.engine";
info.engine_type = EngineType::TENSORRT;
mgr.RegisterModel(info);

// 加载模型
mgr.LoadModel("detector_v1");

// 获取引擎
auto engine = mgr.GetEngine("detector_v1");

// 推理
InferenceResult result;
engine->Infer(input, result);

// 热更新
mgr.HotSwapModel("detector_v1", "detector_v2.engine");

// 统计信息
auto stats = mgr.GetModelStats("detector_v1");
std::cout << "Avg time: " << stats.avg_time_ms << " ms\n";
```

---

## 处理Pipeline

### PreProcessPipelineBuilder

预处理Pipeline构建器。

```cpp
class PreProcessPipelineBuilder {
public:
    PreProcessPipelineBuilder& Resize(int width, int height, bool keep_aspect = true);
    PreProcessPipelineBuilder& Normalize(const std::vector<float>& mean,
                                        const std::vector<float>& std);
    PreProcessPipelineBuilder& NormalizeImageNet();
    PreProcessPipelineBuilder& ColorConvert(...);
    PreProcessPipelineBuilder& HWCToCHW();
    PreProcessPipelineBuilder& Pad(int top, int bottom, int left, int right, float value = 0.0f);

    std::shared_ptr<Pipeline<Tensor, Tensor>> Build();
};
```

**示例:**
```cpp
auto preprocess = PreProcessPipelineBuilder()
    .Resize(640, 640)
    .NormalizeImageNet()
    .HWCToCHW()
    .Build();

Tensor preprocessed;
preprocess->Execute(input, preprocessed);
```

### PostProcessPipelineBuilder

后处理Pipeline构建器。

```cpp
class PostProcessPipelineBuilder {
public:
    PostProcessPipelineBuilder& ApplyNMS(float iou_threshold = 0.5f);
    PostProcessPipelineBuilder& FilterByConfidence(float min_confidence);
    PostProcessPipelineBuilder& SelectTopK(size_t k);

    std::shared_ptr<Pipeline<InferenceResult, InferenceResult>> Build();
};
```

**示例:**
```cpp
auto postprocess = PostProcessPipelineBuilder()
    .FilterByConfidence(0.5f)
    .ApplyNMS(0.45f)
    .SelectTopK(100)
    .Build();

InferenceResult filtered_result;
postprocess->Execute(raw_result, filtered_result);
```

---

## 工具类

### Logger

日志系统。

```cpp
class Logger {
public:
    static Logger& Instance();

    void SetLogLevel(LogLevel level);
    LogLevel GetLogLevel() const;
    void SetEnableConsole(bool enable);
    void SetEnableFile(bool enable);
    void SetLogFile(const std::string& filepath);

    template<typename... Args>
    void Log(LogLevel level, const char* file, int line, Args&&... args);
};

// 日志宏
#define CRKIT_LOG_DEBUG(...)
#define CRKIT_LOG_INFO(...)
#define CRKIT_LOG_WARNING(...)
#define CRKIT_LOG_ERROR(...)
#define CRKIT_LOG_FATAL(...)
```

**示例:**
```cpp
Logger::Instance().SetLogLevel(LogLevel::INFO);

CRKIT_LOG_INFO("Model loaded successfully");
CRKIT_LOG_ERROR("Failed to load model: ", error_msg);
```

### ImageUtils

图像工具类。

```cpp
class ImageUtils {
public:
    static Status LoadImage(const std::string& filepath,
                           Tensor& output,
                           const ImageLoadOptions& options);

    static Status SaveImage(const std::string& filepath,
                           const Tensor& image);

    static Status ResizeImage(const Tensor& input,
                             Tensor& output,
                             int width, int height,
                             bool keep_aspect_ratio = true);

    static Status DrawBoundingBox(Tensor& image,
                                 const BBox& bbox,
                                 const std::string& label = "",
                                 float confidence = 0.0f,
                                 int thickness = 2);

    static Status DrawDetections(Tensor& image,
                                const std::vector<Detection>& detections,
                                int thickness = 2);

    static Status LoadImageBatch(const std::vector<std::string>& filepaths,
                                std::vector<Tensor>& outputs,
                                const ImageLoadOptions& options);

    static Status HWCToCHW(const Tensor& input, Tensor& output);
    static Status CHWToHWC(const Tensor& input, Tensor& output);
};

// 便捷函数
Tensor LoadImage(const std::string& filepath,
                const ImageLoadOptions& options = ImageLoadOptions());
```

**示例:**
```cpp
// 加载图像
ImageLoadOptions options;
options.target_width = 640;
options.target_height = 640;
options.normalize = true;
options.hwc_to_chw = true;

Tensor image = LoadImage("input.jpg", options);

// 绘制检测结果
Tensor result_image = image.Clone();
ImageUtils::DrawDetections(result_image, detections);
ImageUtils::SaveImage("output.jpg", result_image);
```

---

## SDK初始化

```cpp
namespace crkit {
    // 初始化SDK
    Status Initialize();

    // 清理资源
    void Shutdown();

    // 获取版本
    std::string GetVersion();
}
```

**示例:**
```cpp
int main() {
    // 初始化
    crkit::Initialize();

    // ... 使用SDK ...

    // 清理
    crkit::Shutdown();
    return 0;
}
```

---

## 完整示例

```cpp
#include <crkit/crkit.h>

int main() {
    // 1. 初始化
    crkit::Initialize();

    // 2. 配置
    crkit::InferenceConfig config;
    config.device_type = crkit::DeviceType::GPU;
    config.enable_fp16 = true;

    // 3. 创建引擎
    auto engine = crkit::CreateInferenceEngine(
        crkit::EngineType::TENSORRT,
        "model.engine",
        config
    );

    // 4. 加载图像
    crkit::Tensor input = crkit::utils::LoadImage("input.jpg");

    // 5. 推理
    crkit::InferenceResult result;
    engine->Infer(input, result);

    // 6. 处理结果
    result.FilterDetections(0.5f);
    for (const auto& det : result.detections) {
        std::cout << det.label << ": " << det.confidence << "\n";
    }

    // 7. 清理
    crkit::Shutdown();
    return 0;
}
```
