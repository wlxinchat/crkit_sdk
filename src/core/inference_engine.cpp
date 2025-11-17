#include "crkit/engine/inference_engine.h"
#include "crkit/engines/onnxruntime/onnx_engine.h"
#include "crkit/utils/logger.h"

namespace crkit {

std::shared_ptr<IInferenceEngine> InferenceEngineFactory::Create(
    EngineType type,
    const InferenceConfig& config) {

    std::shared_ptr<IInferenceEngine> engine;

    switch (type) {
        case EngineType::ONNXRUNTIME:
            engine = std::make_shared<engines::ONNXEngine>();
            break;

        case EngineType::TENSORRT:
            // TODO: 实现TensorRT引擎
            CRKIT_LOG_ERROR("TensorRT engine not yet implemented");
            return nullptr;

        case EngineType::OPENVINO:
            // TODO: 实现OpenVINO引擎
            CRKIT_LOG_ERROR("OpenVINO engine not yet implemented");
            return nullptr;

        case EngineType::AUTO:
            // 自动选择可用的引擎
            CRKIT_LOG_INFO("Auto-selecting inference engine...");
            // 优先选择ONNX Runtime（跨平台支持最好）
            engine = std::make_shared<engines::ONNXEngine>();
            break;

        default:
            CRKIT_LOG_ERROR("Unsupported engine type");
            return nullptr;
    }

    if (engine) {
        auto status = engine->Initialize(config);
        if (!status.IsOK()) {
            CRKIT_LOG_ERROR("Failed to initialize engine: ", status.Message());
            return nullptr;
        }
    }

    return engine;
}

std::shared_ptr<IInferenceEngine> InferenceEngineFactory::CreateAndLoad(
    EngineType type,
    const std::string& model_path,
    const InferenceConfig& config) {

    auto engine = Create(type, config);
    if (!engine) {
        return nullptr;
    }

    auto status = engine->LoadModel(model_path);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Failed to load model: ", status.Message());
        return nullptr;
    }

    return engine;
}

void InferenceEngineFactory::RegisterEngine(const std::string& name,
                                            EngineCreator creator) {
    // TODO: 实现自定义引擎注册
    CRKIT_LOG_INFO("Registering custom engine: ", name);
}

} // namespace crkit
