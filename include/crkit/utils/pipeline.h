#pragma once

#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"
#include "crkit/core/types.h"
#include <memory>
#include <vector>
#include <functional>

namespace crkit {

// 处理步骤接口
template<typename InputType, typename OutputType>
class IProcessStep {
public:
    virtual ~IProcessStep() = default;
    virtual Status Process(const InputType& input, OutputType& output) = 0;
};

// 图像预处理步骤
namespace preprocessing {

// 调整大小
class Resize : public IProcessStep<Tensor, Tensor> {
public:
    Resize(int width, int height, bool keep_aspect_ratio = true)
        : width_(width), height_(height), keep_aspect_ratio_(keep_aspect_ratio) {}

    Status Process(const Tensor& input, Tensor& output) override;

private:
    int width_;
    int height_;
    bool keep_aspect_ratio_;
};

// 归一化
class Normalize : public IProcessStep<Tensor, Tensor> {
public:
    Normalize(const std::vector<float>& mean,
              const std::vector<float>& std)
        : mean_(mean), std_(std) {}

    // 使用常见的ImageNet均值和标准差
    static Normalize ImageNet() {
        return Normalize({0.485f, 0.456f, 0.406f},
                        {0.229f, 0.224f, 0.225f});
    }

    Status Process(const Tensor& input, Tensor& output) override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

// 颜色空间转换
class ColorConvert : public IProcessStep<Tensor, Tensor> {
public:
    enum class ColorSpace {
        RGB,
        BGR,
        GRAY,
        HSV
    };

    ColorConvert(ColorSpace from, ColorSpace to)
        : from_(from), to_(to) {}

    Status Process(const Tensor& input, Tensor& output) override;

private:
    ColorSpace from_;
    ColorSpace to_;
};

// HWC到CHW转换
class HWCToCHW : public IProcessStep<Tensor, Tensor> {
public:
    Status Process(const Tensor& input, Tensor& output) override;
};

// 填充
class Pad : public IProcessStep<Tensor, Tensor> {
public:
    Pad(int top, int bottom, int left, int right, float value = 0.0f)
        : top_(top), bottom_(bottom), left_(left), right_(right), value_(value) {}

    Status Process(const Tensor& input, Tensor& output) override;

private:
    int top_, bottom_, left_, right_;
    float value_;
};

} // namespace preprocessing

// 后处理步骤
namespace postprocessing {

// NMS (非极大值抑制)
class NMS : public IProcessStep<InferenceResult, InferenceResult> {
public:
    NMS(float iou_threshold = 0.5f) : iou_threshold_(iou_threshold) {}

    Status Process(const InferenceResult& input, InferenceResult& output) override;

private:
    float iou_threshold_;
    float ComputeIOU(const BBox& a, const BBox& b);
};

// 置信度过滤
class ConfidenceFilter : public IProcessStep<InferenceResult, InferenceResult> {
public:
    ConfidenceFilter(float min_confidence) : min_confidence_(min_confidence) {}

    Status Process(const InferenceResult& input, InferenceResult& output) override;

private:
    float min_confidence_;
};

// Top-K选择
class TopK : public IProcessStep<InferenceResult, InferenceResult> {
public:
    TopK(size_t k) : k_(k) {}

    Status Process(const InferenceResult& input, InferenceResult& output) override;

private:
    size_t k_;
};

} // namespace postprocessing

// 处理Pipeline - 链式处理多个步骤
template<typename InputType, typename OutputType>
class Pipeline {
public:
    Pipeline() = default;

    // 添加处理步骤
    template<typename StepType>
    Pipeline& AddStep(std::shared_ptr<StepType> step) {
        steps_.push_back([step](const InputType& in, OutputType& out) {
            return step->Process(in, out);
        });
        return *this;
    }

    // 执行Pipeline
    Status Execute(const InputType& input, OutputType& output) {
        if (steps_.empty()) {
            return Status(StatusCode::ERROR_INVALID_PARAM, "Pipeline is empty");
        }

        // 简化版：假设中间步骤类型相同
        auto current = input;
        for (size_t i = 0; i < steps_.size(); ++i) {
            OutputType temp;
            Status status = steps_[i](current, temp);
            if (!status.IsOK()) {
                return status;
            }
            current = temp;
        }
        output = current;
        return Status();
    }

    // 批量执行
    Status ExecuteBatch(const std::vector<InputType>& inputs,
                       std::vector<OutputType>& outputs) {
        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            Status status = Execute(inputs[i], outputs[i]);
            if (!status.IsOK()) {
                return status;
            }
        }
        return Status();
    }

private:
    using ProcessFunc = std::function<Status(const InputType&, OutputType&)>;
    std::vector<ProcessFunc> steps_;
};

// 预处理Pipeline构建器
class PreProcessPipelineBuilder {
public:
    PreProcessPipelineBuilder& Resize(int width, int height, bool keep_aspect = true);
    PreProcessPipelineBuilder& Normalize(const std::vector<float>& mean,
                                        const std::vector<float>& std);
    PreProcessPipelineBuilder& NormalizeImageNet();
    PreProcessPipelineBuilder& ColorConvert(preprocessing::ColorConvert::ColorSpace from,
                                           preprocessing::ColorConvert::ColorSpace to);
    PreProcessPipelineBuilder& HWCToCHW();
    PreProcessPipelineBuilder& Pad(int top, int bottom, int left, int right, float value = 0.0f);

    std::shared_ptr<Pipeline<Tensor, Tensor>> Build();

private:
    std::shared_ptr<Pipeline<Tensor, Tensor>> pipeline_ =
        std::make_shared<Pipeline<Tensor, Tensor>>();
};

// 后处理Pipeline构建器
class PostProcessPipelineBuilder {
public:
    PostProcessPipelineBuilder& ApplyNMS(float iou_threshold = 0.5f);
    PostProcessPipelineBuilder& FilterByConfidence(float min_confidence);
    PostProcessPipelineBuilder& SelectTopK(size_t k);

    std::shared_ptr<Pipeline<InferenceResult, InferenceResult>> Build();

private:
    std::shared_ptr<Pipeline<InferenceResult, InferenceResult>> pipeline_ =
        std::make_shared<Pipeline<InferenceResult, InferenceResult>>();
};

} // namespace crkit
