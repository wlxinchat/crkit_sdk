#include "crkit/core/inference_context.h"
#include "crkit/utils/logger.h"
#include <algorithm>
#include <limits>

namespace crkit {

Status InferenceContext::Infer(const Tensor& input, InferenceResult& result) {
    if (!engine_) {
        return Status(StatusCode::ERROR_UNKNOWN, "Engine not set");
    }

    Tensor processed_input = input;

    // 预处理
    if (preprocessor_) {
        Tensor temp;
        auto status = preprocessor_->Process(input, temp);
        if (!status.IsOK()) {
            CRKIT_LOG_ERROR("Preprocessing failed: ", status.Message());
            return status;
        }
        processed_input = std::move(temp);
    }

    // 推理
    auto status = engine_->Infer(processed_input, result);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Inference failed: ", status.Message());
        return status;
    }

    // 后处理
    if (postprocessor_) {
        InferenceResult processed_result;
        status = postprocessor_->Process(result.raw_outputs, processed_result);
        if (!status.IsOK()) {
            CRKIT_LOG_ERROR("Postprocessing failed: ", status.Message());
            return status;
        }
        // 保留推理时间
        processed_result.SetInferenceTime(result.GetInferenceTime());
        result = std::move(processed_result);
    }

    // 更新统计
    UpdateStatistics(result.GetInferenceTime());

    return Status();
}

Status InferenceContext::InferBatch(const std::vector<Tensor>& inputs,
                                   std::vector<InferenceResult>& results) {
    results.clear();
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        InferenceResult result;
        auto status = Infer(input, result);
        if (!status.IsOK()) {
            return status;
        }
        results.push_back(std::move(result));
    }

    return Status();
}

Status InferenceContext::InferRaw(const Tensor& input, InferenceResult& result) {
    if (!engine_) {
        return Status(StatusCode::ERROR_UNKNOWN, "Engine not set");
    }

    auto status = engine_->Infer(input, result);
    if (status.IsOK()) {
        UpdateStatistics(result.GetInferenceTime());
    }

    return status;
}

void InferenceContext::ResetStatistics() {
    stats_.total_inferences = 0;
    stats_.total_time_ms = 0.0f;
    stats_.avg_time_ms = 0.0f;
    stats_.min_time_ms = 0.0f;
    stats_.max_time_ms = 0.0f;
}

void InferenceContext::UpdateStatistics(float inference_time) {
    stats_.total_inferences++;
    stats_.total_time_ms += inference_time;
    stats_.avg_time_ms = stats_.total_time_ms / stats_.total_inferences;

    if (stats_.total_inferences == 1) {
        stats_.min_time_ms = inference_time;
        stats_.max_time_ms = inference_time;
    } else {
        stats_.min_time_ms = std::min(stats_.min_time_ms, inference_time);
        stats_.max_time_ms = std::max(stats_.max_time_ms, inference_time);
    }
}

// InferenceContextBuilder实现
InferenceContextBuilder& InferenceContextBuilder::WithEngine(
    EngineType type,
    const std::string& model_path) {
    engine_ = InferenceEngineFactory::CreateAndLoad(type, model_path, config_);
    return *this;
}

InferenceContextBuilder& InferenceContextBuilder::WithEngine(
    std::shared_ptr<IInferenceEngine> engine) {
    engine_ = engine;
    return *this;
}

InferenceContextBuilder& InferenceContextBuilder::WithPreProcessor(
    std::shared_ptr<IPreProcessor> preprocessor) {
    preprocessor_ = preprocessor;
    return *this;
}

InferenceContextBuilder& InferenceContextBuilder::WithPostProcessor(
    std::shared_ptr<IPostProcessor> postprocessor) {
    postprocessor_ = postprocessor;
    return *this;
}

InferenceContextBuilder& InferenceContextBuilder::WithConfig(
    const InferenceConfig& config) {
    config_ = config;
    return *this;
}

std::shared_ptr<InferenceContext> InferenceContextBuilder::Build() {
    auto context = std::make_shared<InferenceContext>();

    if (engine_) {
        context->SetEngine(engine_);
    }

    if (preprocessor_) {
        context->SetPreProcessor(preprocessor_);
    }

    if (postprocessor_) {
        context->SetPostProcessor(postprocessor_);
    }

    return context;
}

} // namespace crkit
