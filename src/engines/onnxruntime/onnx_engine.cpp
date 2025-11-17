#include "crkit/engines/onnxruntime/onnx_engine.h"
#include "crkit/utils/logger.h"
#include <chrono>
#include <cstring>

namespace crkit {
namespace engines {

ONNXEngine::ONNXEngine()
    : inference_count_(0), total_inference_time_(0.0f) {
}

ONNXEngine::~ONNXEngine() {
    Release();
}

Status ONNXEngine::Initialize(const InferenceConfig& config) {
    config_ = config;

    try {
        // 创建ONNX Runtime环境
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CRKITEngine");

        // 设置session选项
        auto status = SetupSessionOptions();
        if (!status.IsOK()) {
            return status;
        }

        CRKIT_LOG_INFO("ONNX Runtime engine initialized successfully");
        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_UNKNOWN,
                     std::string("ONNX Runtime error: ") + e.what());
    }
}

Status ONNXEngine::LoadModel(const std::string& model_path) {
    if (!env_) {
        return Status(StatusCode::ERROR_UNKNOWN, "Engine not initialized");
    }

    try {
        // 加载模型
#ifdef _WIN32
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(),
                                                   *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                                   *session_options_);
#endif

        // 查询模型信息
        auto status = QueryModelInfo();
        if (!status.IsOK()) {
            return status;
        }

        CRKIT_LOG_INFO("Model loaded successfully: ", model_path);
        CRKIT_LOG_INFO("  Inputs: ", input_names_.size());
        CRKIT_LOG_INFO("  Outputs: ", output_names_.size());

        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_MODEL_LOAD_FAILED,
                     std::string("Failed to load model: ") + e.what());
    }
}

Status ONNXEngine::Infer(const Tensor& input, InferenceResult& result) {
    return Infer(std::vector<Tensor>{input}, result);
}

Status ONNXEngine::Infer(const std::vector<Tensor>& inputs,
                        InferenceResult& result) {
    if (!session_) {
        return Status(StatusCode::ERROR_UNKNOWN, "Model not loaded");
    }

    if (inputs.size() != input_names_.size()) {
        return Status(StatusCode::ERROR_INVALID_INPUT,
                     "Input count mismatch");
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 准备输入张量
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names_cstr;

        for (size_t i = 0; i < inputs.size(); ++i) {
            input_tensors.push_back(CreateOrtTensor(inputs[i]));
            input_names_cstr.push_back(input_names_[i].c_str());
        }

        // 准备输出名称
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }

        // 执行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );

        // 转换输出
        result.Clear();
        for (auto& ort_tensor : output_tensors) {
            Tensor output_tensor;
            auto status = ConvertOrtTensorToTensor(ort_tensor, output_tensor);
            if (!status.IsOK()) {
                return status;
            }
            result.raw_outputs.push_back(std::move(output_tensor));
        }

        // 计算推理时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        float inference_time_ms = duration.count() / 1000.0f;

        result.SetInferenceTime(inference_time_ms);

        // 更新统计
        inference_count_++;
        total_inference_time_ += inference_time_ms;

        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_INFERENCE_FAILED,
                     std::string("Inference failed: ") + e.what());
    }
}

Status ONNXEngine::InferBatch(const std::vector<Tensor>& batch_inputs,
                             std::vector<InferenceResult>& results) {
    results.clear();
    results.reserve(batch_inputs.size());

    for (const auto& input : batch_inputs) {
        InferenceResult result;
        auto status = Infer(input, result);
        if (!status.IsOK()) {
            return status;
        }
        results.push_back(std::move(result));
    }

    return Status();
}

Status ONNXEngine::InferAsync(const Tensor& input,
                             InferenceCallback callback) {
    // 简化实现：在单独线程中执行同步推理
    std::thread([this, input, callback]() {
        InferenceResult result;
        Status status = Infer(input, result);
        callback(result, status);
    }).detach();

    return Status();
}

std::vector<std::string> ONNXEngine::GetInputNames() const {
    return input_names_;
}

Shape ONNXEngine::GetInputShape(const std::string& name) const {
    for (size_t i = 0; i < input_names_.size(); ++i) {
        if (input_names_[i] == name) {
            Shape shape;
            for (auto dim : input_shapes_[i]) {
                shape.push_back(dim);
            }
            return shape;
        }
    }
    return Shape();
}

DataType ONNXEngine::GetInputDataType(const std::string& name) const {
    return DataType::FLOAT32;  // 简化实现
}

std::vector<std::string> ONNXEngine::GetOutputNames() const {
    return output_names_;
}

Shape ONNXEngine::GetOutputShape(const std::string& name) const {
    for (size_t i = 0; i < output_names_.size(); ++i) {
        if (output_names_[i] == name) {
            Shape shape;
            for (auto dim : output_shapes_[i]) {
                shape.push_back(dim);
            }
            return shape;
        }
    }
    return Shape();
}

DataType ONNXEngine::GetOutputDataType(const std::string& name) const {
    return DataType::FLOAT32;  // 简化实现
}

std::string ONNXEngine::GetEngineVersion() const {
    return "ONNX Runtime " + std::string(OrtGetApiBase()->GetVersionString());
}

float ONNXEngine::GetAverageInferenceTime() const {
    if (inference_count_ == 0) return 0.0f;
    return total_inference_time_ / inference_count_;
}

void ONNXEngine::ResetStatistics() {
    inference_count_ = 0;
    total_inference_time_ = 0.0f;
}

Status ONNXEngine::WarmUp(int iterations) {
    if (!session_ || input_shapes_.empty()) {
        return Status(StatusCode::ERROR_UNKNOWN, "Model not loaded");
    }

    CRKIT_LOG_INFO("Warming up for ", iterations, " iterations...");

    // 创建虚拟输入
    Tensor dummy_input(Shape(input_shapes_[0].begin(), input_shapes_[0].end()),
                       DataType::FLOAT32);
    dummy_input.Fill(0.0f);

    for (int i = 0; i < iterations; ++i) {
        InferenceResult result;
        auto status = Infer(dummy_input, result);
        if (!status.IsOK()) {
            return status;
        }
    }

    // 重置统计（预热不计入统计）
    ResetStatistics();

    CRKIT_LOG_INFO("Warm-up completed");
    return Status();
}

void ONNXEngine::Release() {
    session_.reset();
    session_options_.reset();
    env_.reset();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
}

Status ONNXEngine::SetupSessionOptions() {
    try {
        session_options_ = std::make_unique<Ort::SessionOptions>();

        // 设置线程数
        session_options_->SetIntraOpNumThreads(config_.num_threads);
        session_options_->SetInterOpNumThreads(config_.num_threads);

        // 设置图优化级别
        session_options_->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU支持
        if (config_.device_type == DeviceType::GPU) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.device_id;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
            CRKIT_LOG_INFO("Using CUDA execution provider on device ",
                          config_.device_id);
#else
            CRKIT_LOG_WARNING("CUDA not available, falling back to CPU");
#endif
        }

        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_UNKNOWN,
                     std::string("Failed to setup session options: ") + e.what());
    }
}

Status ONNXEngine::QueryModelInfo() {
    if (!session_) {
        return Status(StatusCode::ERROR_UNKNOWN, "Session not created");
    }

    try {
        // 查询输入信息
        size_t num_inputs = session_->GetInputCount();
        input_names_.clear();
        input_shapes_.clear();

        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_.push_back(std::string(input_name.get()));

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_.push_back(tensor_info.GetShape());
        }

        // 查询输出信息
        size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        output_shapes_.clear();

        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_.push_back(std::string(output_name.get()));

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(tensor_info.GetShape());
        }

        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_UNKNOWN,
                     std::string("Failed to query model info: ") + e.what());
    }
}

Ort::Value ONNXEngine::CreateOrtTensor(const Tensor& tensor) {
    const auto& shape = tensor.GetShape();
    std::vector<int64_t> ort_shape(shape.begin(), shape.end());

    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(tensor.Data<float>()),
        tensor.NumElements(),
        ort_shape.data(),
        ort_shape.size()
    );
}

Status ONNXEngine::ConvertOrtTensorToTensor(Ort::Value& ort_tensor,
                                           Tensor& tensor) {
    try {
        auto type_info = ort_tensor.GetTensorTypeAndShapeInfo();
        auto ort_shape = type_info.GetShape();

        Shape shape(ort_shape.begin(), ort_shape.end());

        // 创建tensor并拷贝数据
        tensor = Tensor(shape, DataType::FLOAT32);
        float* data = ort_tensor.GetTensorMutableData<float>();
        std::memcpy(tensor.Data(), data, tensor.ByteSize());

        return Status();
    } catch (const Ort::Exception& e) {
        return Status(StatusCode::ERROR_UNKNOWN,
                     std::string("Failed to convert tensor: ") + e.what());
    }
}

DataType ONNXEngine::ONNXTypeToDataType(ONNXTensorElementDataType onnx_type) {
    switch (onnx_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return DataType::FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return DataType::FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return DataType::INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return DataType::INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return DataType::UINT8;
        default:
            return DataType::FLOAT32;
    }
}

} // namespace engines
} // namespace crkit
