#pragma once

#include "crkit/engine/inference_engine.h"
#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"
#include <memory>
#include <functional>

namespace crkit {

// 预处理器接口
class IPreProcessor {
public:
    virtual ~IPreProcessor() = default;

    // 预处理
    virtual Status Process(const Tensor& input, Tensor& output) = 0;

    // 批量预处理
    virtual Status ProcessBatch(const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) = 0;
};

// 后处理器接口
class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;

    // 后处理
    virtual Status Process(const std::vector<Tensor>& model_outputs,
                          InferenceResult& result) = 0;
};

// 推理上下文 - 管理完整的推理流程
class InferenceContext {
public:
    InferenceContext() = default;
    ~InferenceContext() = default;

    // 设置推理引擎
    void SetEngine(std::shared_ptr<IInferenceEngine> engine) {
        engine_ = engine;
    }

    // 设置预处理器
    void SetPreProcessor(std::shared_ptr<IPreProcessor> preprocessor) {
        preprocessor_ = preprocessor;
    }

    // 设置后处理器
    void SetPostProcessor(std::shared_ptr<IPostProcessor> postprocessor) {
        postprocessor_ = postprocessor;
    }

    // 完整推理流程：预处理 -> 推理 -> 后处理
    Status Infer(const Tensor& input, InferenceResult& result);

    // 批量推理
    Status InferBatch(const std::vector<Tensor>& inputs,
                     std::vector<InferenceResult>& results);

    // 仅推理（不做预处理/后处理）
    Status InferRaw(const Tensor& input, InferenceResult& result);

    // 获取推理引擎
    std::shared_ptr<IInferenceEngine> GetEngine() const { return engine_; }

    // 统计信息
    struct Statistics {
        size_t total_inferences = 0;
        float total_time_ms = 0.0f;
        float avg_time_ms = 0.0f;
        float min_time_ms = 0.0f;
        float max_time_ms = 0.0f;
    };

    Statistics GetStatistics() const { return stats_; }
    void ResetStatistics();

private:
    std::shared_ptr<IInferenceEngine> engine_;
    std::shared_ptr<IPreProcessor> preprocessor_;
    std::shared_ptr<IPostProcessor> postprocessor_;
    Statistics stats_;

    void UpdateStatistics(float inference_time);
};

// 推理上下文构建器
class InferenceContextBuilder {
public:
    InferenceContextBuilder& WithEngine(EngineType type, const std::string& model_path);
    InferenceContextBuilder& WithEngine(std::shared_ptr<IInferenceEngine> engine);
    InferenceContextBuilder& WithPreProcessor(std::shared_ptr<IPreProcessor> preprocessor);
    InferenceContextBuilder& WithPostProcessor(std::shared_ptr<IPostProcessor> postprocessor);
    InferenceContextBuilder& WithConfig(const InferenceConfig& config);

    std::shared_ptr<InferenceContext> Build();

private:
    std::shared_ptr<IInferenceEngine> engine_;
    std::shared_ptr<IPreProcessor> preprocessor_;
    std::shared_ptr<IPostProcessor> postprocessor_;
    InferenceConfig config_;
};

} // namespace crkit
