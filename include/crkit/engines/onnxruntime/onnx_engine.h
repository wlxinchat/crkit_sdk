#pragma once

#include "crkit/engine/inference_engine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

namespace crkit {
namespace engines {

// ONNX Runtime推理引擎实现
class ONNXEngine : public IInferenceEngine {
public:
    ONNXEngine();
    ~ONNXEngine() override;

    // 初始化
    Status Initialize(const InferenceConfig& config) override;
    Status LoadModel(const std::string& model_path) override;

    // 推理
    Status Infer(const Tensor& input, InferenceResult& result) override;
    Status Infer(const std::vector<Tensor>& inputs,
                InferenceResult& result) override;
    Status InferBatch(const std::vector<Tensor>& batch_inputs,
                     std::vector<InferenceResult>& results) override;
    Status InferAsync(const Tensor& input,
                     InferenceCallback callback) override;

    // 模型信息
    std::vector<std::string> GetInputNames() const override;
    Shape GetInputShape(const std::string& name) const override;
    DataType GetInputDataType(const std::string& name) const override;
    std::vector<std::string> GetOutputNames() const override;
    Shape GetOutputShape(const std::string& name) const override;
    DataType GetOutputDataType(const std::string& name) const override;

    // 引擎信息
    EngineType GetEngineType() const override { return EngineType::ONNXRUNTIME; }
    std::string GetEngineVersion() const override;

    // 性能统计
    float GetAverageInferenceTime() const override;
    void ResetStatistics() override;
    Status WarmUp(int iterations = 10) override;

    // 资源管理
    void Release() override;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    InferenceConfig config_;

    // 输入输出信息
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // 性能统计
    size_t inference_count_;
    float total_inference_time_;

    // 辅助函数
    Status SetupSessionOptions();
    Status QueryModelInfo();
    Ort::Value CreateOrtTensor(const Tensor& tensor);
    Status ConvertOrtTensorToTensor(Ort::Value& ort_tensor, Tensor& tensor);
    DataType ONNXTypeToDataType(ONNXTensorElementDataType onnx_type);
};

} // namespace engines
} // namespace crkit
