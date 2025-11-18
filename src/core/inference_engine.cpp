#include "crkit/engine/inference_engine.h"
#include "crkit/engines/onnxruntime/onnx_engine.h"
#include "crkit/utils/logger.h"
#include <unordered_map>

namespace crkit {

// 引擎注册表（静态全局）
static std::unordered_map<std::string, InferenceEngineFactory::EngineCreator>& GetEngineRegistry() {
    static std::unordered_map<std::string, InferenceEngineFactory::EngineCreator> registry;
    return registry;
}

std::shared_ptr<IInferenceEngine> InferenceEngineFactory::Create(
    EngineType type,
    const InferenceConfig& config) {

    std::shared_ptr<IInferenceEngine> engine;

    switch (type) {
        case EngineType::ONNXRUNTIME:
            engine = std::make_shared<engines::ONNXEngine>();
            break;

        case EngineType::TENSORRT:
            CRKIT_LOG_WARNING("TensorRT engine not yet implemented");
            return nullptr;

        case EngineType::OPENVINO:
            CRKIT_LOG_WARNING("OpenVINO engine not yet implemented");
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

std::shared_ptr<IInferenceEngine> InferenceEngineFactory::CreateByName(
    const std::string& name,
    const InferenceConfig& config) {

    auto& registry = GetEngineRegistry();
    auto it = registry.find(name);

    if (it == registry.end()) {
        CRKIT_LOG_ERROR("Custom engine not found: ", name);
        return nullptr;
    }

    CRKIT_LOG_INFO("Creating custom engine: ", name);

    auto engine = it->second();
    if (!engine) {
        CRKIT_LOG_ERROR("Failed to create custom engine: ", name);
        return nullptr;
    }

    auto status = engine->Initialize(config);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Failed to initialize custom engine ", name, ": ", status.Message());
        return nullptr;
    }

    return engine;
}

void InferenceEngineFactory::RegisterEngine(const std::string& name,
                                            EngineCreator creator) {
    if (name.empty()) {
        CRKIT_LOG_ERROR("Engine name cannot be empty");
        return;
    }

    if (!creator) {
        CRKIT_LOG_ERROR("Engine creator cannot be null");
        return;
    }

    auto& registry = GetEngineRegistry();

    if (registry.find(name) != registry.end()) {
        CRKIT_LOG_WARNING("Overwriting existing engine: ", name);
    }

    registry[name] = std::move(creator);
    CRKIT_LOG_INFO("Successfully registered custom engine: ", name);
}

std::vector<std::string> InferenceEngineFactory::GetRegisteredEngines() {
    auto& registry = GetEngineRegistry();
    std::vector<std::string> names;
    names.reserve(registry.size());

    for (const auto& pair : registry) {
        names.push_back(pair.first);
    }

    return names;
}

} // namespace crkit
