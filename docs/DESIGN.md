# CRKIT SDK 设计文档

## 1. 设计目标

CRKIT SDK 是一个专为工业质检场景设计的通用AI推理SDK，旨在提供：

- **统一接口**: 支持多种推理引擎（TensorRT、ONNX Runtime、OpenVINO等），用户无需关心底层实现
- **高性能**: 针对工业场景优化，支持批处理、异步推理、内存池等性能优化技术
- **易用性**: 简洁的API设计，最小化集成成本
- **灵活性**: 支持自定义预处理/后处理Pipeline，适应不同质检场景
- **可靠性**: 完善的错误处理、日志系统、资源管理

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (User Code)                    │
├─────────────────────────────────────────────────────────┤
│                    高层API                               │
│  - InferenceContext (推理上下文)                        │
│  - ModelManager (模型管理器)                            │
│  - Pipeline (预处理/后处理)                             │
├─────────────────────────────────────────────────────────┤
│                    核心抽象层                            │
│  - IInferenceEngine (推理引擎接口)                      │
│  - IPreProcessor (预处理接口)                           │
│  - IPostProcessor (后处理接口)                          │
├─────────────────────────────────────────────────────────┤
│                    引擎实现层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │TensorRT  │  │ONNX RT   │  │OpenVINO  │              │
│  │ Engine   │  │ Engine   │  │ Engine   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
├─────────────────────────────────────────────────────────┤
│                    硬件加速层                            │
│       CUDA / cuDNN / TensorRT / MKL-DNN / ...          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 数据结构层 (data/)

**Tensor**
- 统一的张量表示，支持多种数据类型（FP32/FP16/INT8等）
- 智能指针管理内存，支持零拷贝
- 提供形状变换、克隆等基本操作

**InferenceResult**
- 统一的推理结果表示
- 支持多种任务类型：检测(Detection)、分类(Classification)、分割(Segmentation)
- 提供结果过滤、排序等便捷操作

#### 2.2.2 引擎抽象层 (engine/)

**IInferenceEngine**
- 推理引擎的抽象接口
- 定义了统一的推理API：
  - 单输入/多输入推理
  - 批量推理
  - 异步推理
  - 模型信息查询
- 工厂模式创建具体引擎实现

**InferenceConfig**
- 统一的推理配置
- 支持引擎类型、设备选择、精度设置等

#### 2.2.3 核心功能层 (core/)

**InferenceContext**
- 推理上下文，管理完整的推理流程
- 整合预处理、推理、后处理
- 提供统计信息收集

**ModelManager**
- 单例模式的模型管理器
- 支持多模型注册、加载、卸载
- 支持模型热更新
- LRU策略自动管理内存

**Types**
- 基础类型定义
- 统一的错误码和状态管理

#### 2.2.4 处理Pipeline (utils/pipeline.h)

**预处理Pipeline**
- 图像resize、normalize、颜色空间转换
- 支持链式调用
- 可扩展的处理步骤

**后处理Pipeline**
- NMS（非极大值抑制）
- 置信度过滤
- Top-K选择

#### 2.2.5 工具类 (utils/)

**Logger**
- 线程安全的日志系统
- 支持多级别日志
- 支持控制台和文件输出

**ImageUtils**
- 图像加载、保存
- 图像预处理
- 结果可视化

## 3. 关键设计决策

### 3.1 接口设计原则

1. **抽象与实现分离**: 用户只需要关心接口，不需要了解具体引擎实现
2. **最小惊讶原则**: API行为符合直觉，命名清晰
3. **资源安全**: 使用RAII和智能指针自动管理资源
4. **错误处理**: 使用Status类统一错误处理，避免异常

### 3.2 性能优化

1. **零拷贝**: Tensor支持外部内存，避免不必要的数据拷贝
2. **批处理**: 支持批量推理，提高GPU利用率
3. **异步推理**: 支持异步API，提高吞吐量
4. **内存池**: 可选的内存池管理，减少内存分配开销
5. **预热**: 提供WarmUp接口，预先分配资源

### 3.3 工业场景适配

1. **多模型支持**: ModelManager支持同时管理多个模型
2. **热更新**: 支持不停机更新模型
3. **Pipeline灵活性**: 支持自定义预处理/后处理步骤
4. **结果类型丰富**: 支持检测、分类、分割等多种任务

### 3.4 扩展性设计

1. **插件化引擎**: 通过工厂模式注册新引擎
2. **自定义处理步骤**: Pipeline支持添加自定义步骤
3. **回调机制**: 异步推理支持回调函数

## 4. 使用流程

### 4.1 基础使用流程

```cpp
// 1. 初始化SDK
crkit::Initialize();

// 2. 创建推理引擎
auto engine = crkit::CreateInferenceEngine(
    crkit::EngineType::TENSORRT,
    "model.engine"
);

// 3. 加载图像
crkit::Tensor input = crkit::utils::LoadImage("image.jpg");

// 4. 执行推理
crkit::InferenceResult result;
engine->Infer(input, result);

// 5. 处理结果
for (const auto& det : result.detections) {
    // 处理检测结果
}

// 6. 清理资源
crkit::Shutdown();
```

### 4.2 高级使用流程（Pipeline）

```cpp
// 1. 构建预处理Pipeline
auto preprocess = crkit::PreProcessPipelineBuilder()
    .Resize(640, 640)
    .NormalizeImageNet()
    .HWCToCHW()
    .Build();

// 2. 构建后处理Pipeline
auto postprocess = crkit::PostProcessPipelineBuilder()
    .FilterByConfidence(0.5f)
    .ApplyNMS(0.45f)
    .Build();

// 3. 创建推理上下文
auto context = crkit::InferenceContextBuilder()
    .WithEngine(engine)
    .WithPreProcessor(preprocess)
    .WithPostProcessor(postprocess)
    .Build();

// 4. 执行完整推理流程
crkit::InferenceResult result;
context->Infer(input, result);
```

## 5. 线程安全性

- **ModelManager**: 线程安全，使用mutex保护内部状态
- **IInferenceEngine**: 具体实现需要保证线程安全
- **Tensor**: 线程不安全，需要外部同步
- **Pipeline**: 无状态，天然线程安全

## 6. 性能基准

预期性能指标（参考值）：

| 场景 | 分辨率 | 引擎 | 设备 | 延迟 | 吞吐量 |
|------|--------|------|------|------|--------|
| 目标检测 | 640x640 | TensorRT | RTX 3090 | ~3ms | 300+ fps |
| 目标检测 | 640x640 | ONNX RT | CPU (16核) | ~30ms | 30+ fps |
| 分类 | 224x224 | TensorRT | RTX 3090 | ~1ms | 1000+ fps |

## 7. 未来扩展

### 7.1 短期规划
- [ ] 添加更多预处理/后处理操作
- [ ] 支持动态batch size
- [ ] 添加性能分析工具
- [ ] 完善错误处理和日志

### 7.2 长期规划
- [ ] 支持模型量化和优化
- [ ] 添加分布式推理支持
- [ ] 支持更多推理引擎（NCNN、MNN等）
- [ ] 提供Python绑定
- [ ] 添加模型转换工具

## 8. 最佳实践

### 8.1 性能优化建议

1. **批量处理**: 尽可能使用批量推理
2. **预热**: 首次推理前调用WarmUp
3. **FP16**: 在GPU上启用FP16精度
4. **固定输入尺寸**: 避免动态尺寸带来的性能损失

### 8.2 内存管理建议

1. **及时释放**: 不使用的模型及时卸载
2. **LRU策略**: 设置合理的max_loaded_models
3. **零拷贝**: 对大数据使用零拷贝

### 8.3 错误处理建议

1. **检查返回状态**: 总是检查Status返回值
2. **日志记录**: 记录关键操作的日志
3. **异常恢复**: 提供降级策略

## 9. 依赖项

### 9.1 核心依赖
- C++17标准库
- CMake 3.16+

### 9.2 可选依赖
- CUDA 11.0+ (GPU支持)
- TensorRT 8.0+ (TensorRT引擎)
- ONNX Runtime 1.10+ (ONNX引擎)
- OpenVINO 2022+ (OpenVINO引擎)
- OpenCV 4.0+ (图像处理)

## 10. 总结

CRKIT SDK通过清晰的分层架构、统一的接口设计和灵活的扩展机制，为工业质检场景提供了一个高性能、易用的AI推理解决方案。设计遵循SOLID原则，确保代码的可维护性和可扩展性。
