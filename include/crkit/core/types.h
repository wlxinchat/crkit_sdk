#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace crkit {

// 数据类型
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    UINT8
};

// 推理引擎类型
enum class EngineType {
    TENSORRT,      // NVIDIA TensorRT
    ONNXRUNTIME,   // ONNX Runtime
    OPENVINO,      // Intel OpenVINO
    NCNN,          // Tencent NCNN
    AUTO           // 自动选择最优引擎
};

// 设备类型
enum class DeviceType {
    CPU,
    GPU,
    AUTO
};

// 推理模式
enum class InferenceMode {
    SYNC,          // 同步推理
    ASYNC          // 异步推理
};

// 错误码
enum class StatusCode {
    SUCCESS = 0,
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

// 状态类
class Status {
public:
    Status() : code_(StatusCode::SUCCESS) {}
    Status(StatusCode code, const std::string& msg = "")
        : code_(code), message_(msg) {}

    bool IsOK() const { return code_ == StatusCode::SUCCESS; }
    StatusCode Code() const { return code_; }
    const std::string& Message() const { return message_; }

    operator bool() const { return IsOK(); }

private:
    StatusCode code_;
    std::string message_;
};

// 张量形状
using Shape = std::vector<int64_t>;

// 计算元素总数
inline int64_t ShapeSize(const Shape& shape) {
    if (shape.empty()) return 0;
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// 获取数据类型大小
inline size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32:   return 4;
        case DataType::INT8:    return 1;
        case DataType::UINT8:   return 1;
        default:                return 0;
    }
}

} // namespace crkit
