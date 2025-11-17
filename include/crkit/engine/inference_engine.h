#pragma once

#include "crkit/core/types.h"
#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace crkit {

// 推理配置
struct InferenceConfig {
    EngineType engine_type = EngineType::AUTO;   // 推理引擎类型
    DeviceType device_type = DeviceType::AUTO;   // 设备类型
    int device_id = 0;                            // 设备ID（GPU编号）
    int max_batch_size = 1;                       // 最大批次大小
    int num_threads = 4;                          // CPU线程数
    bool enable_fp16 = false;                     // 启用FP16精度
    bool enable_profiling = false;                // 启用性能分析
    InferenceMode mode = InferenceMode::SYNC;     // 推理模式

    // 内存优化
    size_t workspace_size = 1 << 30;              // 工作空间大小（1GB）
    bool use_memory_pool = true;                  // 使用内存池

    // 模型相关
    std::string model_path;                       // 模型文件路径
    std::map<std::string, std::string> metadata; // 额外配置
};

// 推理引擎接口（抽象基类）
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // 初始化引擎
    virtual Status Initialize(const InferenceConfig& config) = 0;

    // 加载模型
    virtual Status LoadModel(const std::string& model_path) = 0;

    // 推理（单输入）
    virtual Status Infer(const Tensor& input, InferenceResult& result) = 0;

    // 推理（多输入）
    virtual Status Infer(const std::vector<Tensor>& inputs,
                        InferenceResult& result) = 0;

    // 批量推理
    virtual Status InferBatch(const std::vector<Tensor>& batch_inputs,
                             std::vector<InferenceResult>& results) = 0;

    // 异步推理
    using InferenceCallback = std::function<void(const InferenceResult&, Status)>;
    virtual Status InferAsync(const Tensor& input,
                             InferenceCallback callback) = 0;

    // 获取输入信息
    virtual std::vector<std::string> GetInputNames() const = 0;
    virtual Shape GetInputShape(const std::string& name) const = 0;
    virtual DataType GetInputDataType(const std::string& name) const = 0;

    // 获取输出信息
    virtual std::vector<std::string> GetOutputNames() const = 0;
    virtual Shape GetOutputShape(const std::string& name) const = 0;
    virtual DataType GetOutputDataType(const std::string& name) const = 0;

    // 获取引擎信息
    virtual EngineType GetEngineType() const = 0;
    virtual std::string GetEngineVersion() const = 0;

    // 性能统计
    virtual float GetAverageInferenceTime() const = 0;
    virtual void ResetStatistics() = 0;

    // 资源管理
    virtual Status WarmUp(int iterations = 10) = 0;  // 预热
    virtual void Release() = 0;                       // 释放资源
};

// 推理引擎工厂
class InferenceEngineFactory {
public:
    // 创建推理引擎
    static std::shared_ptr<IInferenceEngine> Create(
        EngineType type,
        const InferenceConfig& config = InferenceConfig());

    // 创建推理引擎并加载模型
    static std::shared_ptr<IInferenceEngine> CreateAndLoad(
        EngineType type,
        const std::string& model_path,
        const InferenceConfig& config = InferenceConfig());

    // 注册自定义引擎
    using EngineCreator = std::function<std::shared_ptr<IInferenceEngine>()>;
    static void RegisterEngine(const std::string& name, EngineCreator creator);

private:
    InferenceEngineFactory() = default;
};

// 便捷创建函数
inline std::shared_ptr<IInferenceEngine> CreateInferenceEngine(
    EngineType type,
    const std::string& model_path,
    const InferenceConfig& config = InferenceConfig()) {
    return InferenceEngineFactory::CreateAndLoad(type, model_path, config);
}

} // namespace crkit
