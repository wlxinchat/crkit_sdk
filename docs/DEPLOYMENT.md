# CRKIT SDK 部署指南

本文档介绍如何在生产环境中部署CRKIT SDK。

## 目录
- [部署架构](#部署架构)
- [Docker部署](#docker部署)
- [性能优化](#性能优化)
- [监控和日志](#监控和日志)
- [故障排查](#故障排查)

---

## 部署架构

### 典型工业质检系统架构

```
┌─────────────────────────────────────────────────────┐
│                    图像采集层                        │
│    工业相机 → 图像缓存 → 预处理模块                 │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                  CRKIT推理层                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  模型加载  │  │  批量推理  │  │  结果处理  │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                    结果处理层                        │
│    质量判定 → 数据存储 → 报警通知                   │
└─────────────────────────────────────────────────────┘
```

---

## Docker部署

### Dockerfile

创建 `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ \
    && ldconfig \
    && rm -rf onnxruntime-linux-x64-1.16.3*

# 拷贝SDK源码
WORKDIR /app
COPY . /app/crkit_sdk

# 编译SDK
RUN cd /app/crkit_sdk && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_EXAMPLES=ON \
             -DENABLE_ONNXRUNTIME=ON && \
    make -j$(nproc) && \
    make install

# 设置工作目录
WORKDIR /app

# 暴露端口（如果有API服务）
EXPOSE 8080

# 启动命令
CMD ["/bin/bash"]
```

### docker-compose.yml

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  crkit_inference:
    build: .
    image: crkit_sdk:latest
    container_name: crkit_inference
    volumes:
      - ./models:/app/models
      - ./images:/app/images
      - ./results:/app/results
    environment:
      - OMP_NUM_THREADS=4
      - CRKIT_LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped

  # GPU版本（需要nvidia-docker）
  crkit_inference_gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    image: crkit_sdk:gpu
    container_name: crkit_inference_gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
      - ./images:/app/images
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 性能优化

### 1. 推理引擎优化

#### ONNX Runtime优化

```cpp
crkit::InferenceConfig config;

// CPU优化
config.num_threads = std::thread::hardware_concurrency();
config.device_type = crkit::DeviceType::CPU;

// GPU优化（如果有NVIDIA GPU）
config.device_type = crkit::DeviceType::GPU;
config.device_id = 0;
config.enable_fp16 = true;  // 启用FP16精度
```

#### TensorRT优化（最佳性能）

```cpp
// 转换ONNX模型为TensorRT引擎
// 使用trtexec工具
$ trtexec --onnx=yolo11n.onnx \
          --saveEngine=yolo11n.engine \
          --fp16 \
          --workspace=4096 \
          --minShapes=input:1x3x640x640 \
          --optShapes=input:1x3x640x640 \
          --maxShapes=input:8x3x640x640

// 使用TensorRT引擎
config.engine_type = crkit::EngineType::TENSORRT;
config.enable_fp16 = true;
config.max_batch_size = 8;
```

### 2. 批处理优化

对于高吞吐量场景，使用批量推理：

```cpp
// 累积图像到批次
std::vector<crkit::Tensor> batch_inputs;
const int batch_size = 8;

for (const auto& image_path : image_paths) {
    auto input = LoadAndPreprocess(image_path);
    batch_inputs.push_back(input);

    if (batch_inputs.size() == batch_size) {
        // 批量推理
        std::vector<crkit::InferenceResult> results;
        engine->InferBatch(batch_inputs, results);

        // 处理结果...

        batch_inputs.clear();
    }
}
```

### 3. 内存优化

```cpp
// 限制同时加载的模型数量
auto& mgr = crkit::ModelManager::Instance();
mgr.SetMaxLoadedModels(3);

// 定期清理长时间未使用的模型
mgr.CleanupUnusedModels(300);  // 5分钟

// 使用零拷贝
crkit::Tensor tensor(external_data_ptr, shape, dtype, false);  // copy=false
```

### 4. 多线程并发

```cpp
// 创建线程池处理多路相机输入
#include <thread>
#include <queue>
#include <mutex>

class InferenceWorker {
public:
    void Start(int num_workers) {
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&InferenceWorker::WorkerThread, this, i);
        }
    }

private:
    void WorkerThread(int worker_id) {
        // 每个worker使用独立的模型实例
        auto engine = LoadEngine(worker_id);

        while (running_) {
            auto task = GetNextTask();
            if (task) {
                ProcessTask(task, engine);
            }
        }
    }

    std::vector<std::thread> workers_;
    bool running_ = true;
};
```

---

## 监控和日志

### 1. 日志配置

```cpp
// 配置日志级别
crkit::Logger::Instance().SetLogLevel(crkit::LogLevel::INFO);

// 启用文件日志
crkit::Logger::Instance().SetEnableFile(true);
crkit::Logger::Instance().SetLogFile("/var/log/crkit/inference.log");
```

### 2. 性能监控

```cpp
class PerformanceMonitor {
public:
    void RecordInference(float time_ms, int num_detections) {
        std::lock_guard<std::mutex> lock(mutex_);

        stats_.total_inferences++;
        stats_.total_time += time_ms;
        stats_.total_detections += num_detections;

        // 每1000次推理输出统计
        if (stats_.total_inferences % 1000 == 0) {
            PrintStatistics();
        }
    }

    void PrintStatistics() {
        float avg_time = stats_.total_time / stats_.total_inferences;
        float fps = 1000.0f / avg_time;
        float avg_detections = static_cast<float>(stats_.total_detections) /
                               stats_.total_inferences;

        CRKIT_LOG_INFO("Performance Stats:");
        CRKIT_LOG_INFO("  Total inferences: ", stats_.total_inferences);
        CRKIT_LOG_INFO("  Average time: ", avg_time, " ms");
        CRKIT_LOG_INFO("  FPS: ", fps);
        CRKIT_LOG_INFO("  Avg detections: ", avg_detections);
    }

private:
    struct Stats {
        size_t total_inferences = 0;
        float total_time = 0.0f;
        size_t total_detections = 0;
    } stats_;

    std::mutex mutex_;
};
```

### 3. 系统监控指标

监控以下关键指标：

- **推理延迟**: 单次推理耗时
- **吞吐量**: 每秒处理图像数
- **内存占用**: 进程内存使用量
- **GPU利用率**: GPU使用率（如果使用GPU）
- **检测率**: 缺陷检出率
- **误报率**: 误检率

### 4. 集成Prometheus监控

```cpp
// 暴露Prometheus metrics端点
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>

class MetricsCollector {
public:
    MetricsCollector(const std::string& endpoint = "0.0.0.0:8080")
        : exposer_(endpoint) {

        auto registry = std::make_shared<prometheus::Registry>();

        // 创建指标
        auto& inference_counter = prometheus::BuildCounter()
            .Name("crkit_inference_total")
            .Help("Total number of inferences")
            .Register(*registry);

        inference_total_ = &inference_counter.Add({});

        auto& inference_duration = prometheus::BuildHistogram()
            .Name("crkit_inference_duration_ms")
            .Help("Inference duration in milliseconds")
            .Register(*registry);

        inference_duration_ = &inference_duration.Add({},
            prometheus::Histogram::BucketBoundaries{1, 5, 10, 25, 50, 100, 250, 500});

        exposer_.RegisterCollectable(registry);
    }

    void RecordInference(float duration_ms) {
        inference_total_->Increment();
        inference_duration_->Observe(duration_ms);
    }

private:
    prometheus::Exposer exposer_;
    prometheus::Counter* inference_total_;
    prometheus::Histogram* inference_duration_;
};
```

---

## 故障排查

### 常见问题和解决方案

#### 1. 推理失败

**症状**: 返回ERROR_INFERENCE_FAILED

**检查步骤**:
```bash
# 1. 检查模型文件是否完整
md5sum model.onnx

# 2. 检查输入维度是否正确
# 使用netron查看模型
pip install netron
netron model.onnx

# 3. 检查ONNX Runtime版本
ldd libcrkit_onnxruntime.so | grep onnxruntime

# 4. 启用详细日志
export CRKIT_LOG_LEVEL=DEBUG
```

#### 2. 内存泄漏

**症状**: 长时间运行后内存持续增长

**检查步骤**:
```bash
# 使用valgrind检测
valgrind --leak-check=full --show-leak-kinds=all ./your_app

# 使用gperftools
LD_PRELOAD=/usr/lib/libtcmalloc.so HEAPPROFILE=/tmp/heap.prof ./your_app
google-pprof --pdf ./your_app /tmp/heap.prof.0001.heap > heap.pdf
```

**常见原因**:
- 未调用`engine->Release()`
- 模型未卸载
- Tensor对象未释放

#### 3. 性能下降

**症状**: 推理速度远低于预期

**检查清单**:
- [ ] 确认使用Release模式编译
- [ ] 检查CPU/GPU利用率
- [ ] 验证批处理是否生效
- [ ] 检查是否有I/O瓶颈
- [ ] 验证预热是否执行

```cpp
// 性能分析
engine->WarmUp(100);  // 充分预热

auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 1000; ++i) {
    engine->Infer(input, result);
}
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Avg time: " << duration.count() / 1000.0 << " ms" << std::endl;
```

#### 4. GPU内存不足

**症状**: CUDA out of memory错误

**解决方案**:
```cpp
// 1. 减小batch size
config.max_batch_size = 1;

// 2. 限制GPU内存增长
// ONNX Runtime CUDA选项
// setenv("ORT_CUDA_MEM_LIMIT", "2GB", 1);

// 3. 清理不使用的模型
ModelManager::Instance().UnloadModel("unused_model");
```

---

## 生产环境最佳实践

### 1. 配置管理

使用配置文件而非硬编码：

```cpp
// config.json
{
  "models": [
    {
      "id": "defect_detector",
      "path": "/models/yolo11_defect.onnx",
      "device": "gpu",
      "batch_size": 4
    }
  ],
  "inference": {
    "num_threads": 4,
    "conf_threshold": 0.5,
    "nms_threshold": 0.45
  }
}

// 加载配置
auto config = LoadConfigFromFile("config.json");
```

### 2. 优雅关闭

```cpp
// 信号处理
#include <csignal>

std::atomic<bool> running(true);

void SignalHandler(int signum) {
    CRKIT_LOG_INFO("Received signal ", signum, ", shutting down...");
    running = false;
}

int main() {
    // 注册信号处理器
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    crkit::Initialize();

    while (running) {
        // 处理任务...
    }

    // 清理资源
    crkit::ModelManager::Instance().UnloadAllModels();
    crkit::Shutdown();

    return 0;
}
```

### 3. 错误恢复

```cpp
// 推理失败重试机制
Status InferWithRetry(IInferenceEngine* engine,
                     const Tensor& input,
                     InferenceResult& result,
                     int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        auto status = engine->Infer(input, result);

        if (status.IsOK()) {
            return status;
        }

        CRKIT_LOG_WARNING("Inference failed, retry ", i + 1, "/", max_retries);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return Status(StatusCode::ERROR_INFERENCE_FAILED,
                 "Max retries exceeded");
}
```

### 4. 健康检查

```cpp
// HTTP健康检查端点
bool HealthCheck() {
    try {
        // 检查模型是否加载
        if (!ModelManager::Instance().IsModelLoaded("main_model")) {
            return false;
        }

        // 测试推理
        crkit::Tensor dummy_input({1, 3, 640, 640}, crkit::DataType::FLOAT32);
        crkit::InferenceResult result;

        auto engine = ModelManager::Instance().GetEngine("main_model");
        auto status = engine->Infer(dummy_input, result);

        return status.IsOK();
    } catch (...) {
        return false;
    }
}
```

---

## 总结

部署CRKIT SDK时的关键要点：

1. **选择合适的推理引擎**: ONNX Runtime (通用) vs TensorRT (高性能)
2. **性能优化**: 批处理、多线程、GPU加速
3. **资源管理**: 内存限制、模型缓存、定期清理
4. **监控告警**: 性能指标、错误率、资源使用
5. **故障恢复**: 重试机制、健康检查、优雅关闭

有问题欢迎提Issue或联系技术支持！
