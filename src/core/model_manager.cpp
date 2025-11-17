#include "crkit/core/model_manager.h"
#include "crkit/utils/logger.h"
#include <algorithm>
#include <chrono>

namespace crkit {

ModelManager& ModelManager::Instance() {
    static ModelManager instance;
    return instance;
}

Status ModelManager::RegisterModel(const ModelInfo& model_info) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (models_.find(model_info.model_id) != models_.end()) {
        CRKIT_LOG_WARNING("Model already registered: ", model_info.model_id);
    }

    ModelEntry entry;
    entry.info = model_info;
    entry.is_loaded = false;
    entry.last_access_time = 0;
    entry.stats.model_id = model_info.model_id;

    models_[model_info.model_id] = entry;

    CRKIT_LOG_INFO("Model registered: ", model_info.model_id);
    return Status();
}

Status ModelManager::LoadModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return Status(StatusCode::ERROR_INVALID_PARAM,
                     "Model not registered: " + model_id);
    }

    if (it->second.is_loaded) {
        CRKIT_LOG_INFO("Model already loaded: ", model_id);
        UpdateAccessTime(model_id);
        return Status();
    }

    // 检查是否需要卸载其他模型
    size_t loaded_count = 0;
    for (const auto& pair : models_) {
        if (pair.second.is_loaded) loaded_count++;
    }

    if (loaded_count >= max_loaded_models_) {
        EvictLRUModel();
    }

    // 创建推理引擎
    it->second.engine = InferenceEngineFactory::CreateAndLoad(
        it->second.info.engine_type,
        it->second.info.model_path,
        it->second.info.config
    );

    if (!it->second.engine) {
        return Status(StatusCode::ERROR_MODEL_LOAD_FAILED,
                     "Failed to create engine for: " + model_id);
    }

    // 创建推理上下文
    it->second.context = std::make_shared<InferenceContext>();
    it->second.context->SetEngine(it->second.engine);

    it->second.is_loaded = true;
    UpdateAccessTime(model_id);

    CRKIT_LOG_INFO("Model loaded successfully: ", model_id);
    return Status();
}

Status ModelManager::UnloadModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return Status(StatusCode::ERROR_INVALID_PARAM,
                     "Model not found: " + model_id);
    }

    if (!it->second.is_loaded) {
        return Status();
    }

    it->second.engine->Release();
    it->second.engine.reset();
    it->second.context.reset();
    it->second.is_loaded = false;

    CRKIT_LOG_INFO("Model unloaded: ", model_id);
    return Status();
}

std::shared_ptr<IInferenceEngine> ModelManager::GetEngine(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end() || !it->second.is_loaded) {
        CRKIT_LOG_ERROR("Model not loaded: ", model_id);
        return nullptr;
    }

    UpdateAccessTime(model_id);
    return it->second.engine;
}

std::shared_ptr<InferenceContext> ModelManager::GetContext(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end() || !it->second.is_loaded) {
        CRKIT_LOG_ERROR("Model not loaded: ", model_id);
        return nullptr;
    }

    UpdateAccessTime(model_id);
    return it->second.context;
}

bool ModelManager::IsModelLoaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    return it != models_.end() && it->second.is_loaded;
}

ModelInfo ModelManager::GetModelInfo(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it != models_.end()) {
        return it->second.info;
    }
    return ModelInfo();
}

std::vector<std::string> ModelManager::ListRegisteredModels() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> models;
    for (const auto& pair : models_) {
        models.push_back(pair.first);
    }
    return models;
}

std::vector<std::string> ModelManager::ListLoadedModels() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> models;
    for (const auto& pair : models_) {
        if (pair.second.is_loaded) {
            models.push_back(pair.first);
        }
    }
    return models;
}

Status ModelManager::HotSwapModel(const std::string& model_id,
                                 const std::string& new_model_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return Status(StatusCode::ERROR_INVALID_PARAM,
                     "Model not found: " + model_id);
    }

    // 保存旧引擎
    auto old_engine = it->second.engine;

    // 创建新引擎
    auto new_engine = InferenceEngineFactory::CreateAndLoad(
        it->second.info.engine_type,
        new_model_path,
        it->second.info.config
    );

    if (!new_engine) {
        return Status(StatusCode::ERROR_MODEL_LOAD_FAILED,
                     "Failed to load new model");
    }

    // 更新引擎
    it->second.engine = new_engine;
    it->second.context->SetEngine(new_engine);
    it->second.info.model_path = new_model_path;

    // 释放旧引擎
    if (old_engine) {
        old_engine->Release();
    }

    CRKIT_LOG_INFO("Model hot-swapped: ", model_id);
    return Status();
}

Status ModelManager::LoadModels(const std::vector<std::string>& model_ids) {
    for (const auto& model_id : model_ids) {
        auto status = LoadModel(model_id);
        if (!status.IsOK()) {
            CRKIT_LOG_ERROR("Failed to load model: ", model_id);
            return status;
        }
    }
    return Status();
}

void ModelManager::UnloadAllModels() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : models_) {
        if (pair.second.is_loaded && pair.second.engine) {
            pair.second.engine->Release();
            pair.second.engine.reset();
            pair.second.context.reset();
            pair.second.is_loaded = false;
        }
    }

    CRKIT_LOG_INFO("All models unloaded");
}

void ModelManager::CleanupUnusedModels(int64_t idle_time_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    int64_t threshold = idle_time_seconds * 1000000000LL;  // Convert to nanoseconds

    std::vector<std::string> models_to_unload;

    for (const auto& pair : models_) {
        if (pair.second.is_loaded) {
            int64_t idle_time = now - pair.second.last_access_time;
            if (idle_time > threshold) {
                models_to_unload.push_back(pair.first);
            }
        }
    }

    for (const auto& model_id : models_to_unload) {
        UnloadModel(model_id);
        CRKIT_LOG_INFO("Unloaded idle model: ", model_id);
    }
}

ModelManager::ModelStats ModelManager::GetModelStats(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it != models_.end()) {
        return it->second.stats;
    }
    return ModelStats();
}

void ModelManager::SetMaxLoadedModels(size_t max_models) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_loaded_models_ = max_models;
}

void ModelManager::EvictLRUModel() {
    // 找到最久未使用的模型
    std::string lru_model_id;
    int64_t oldest_time = std::numeric_limits<int64_t>::max();

    for (const auto& pair : models_) {
        if (pair.second.is_loaded &&
            pair.second.last_access_time < oldest_time) {
            oldest_time = pair.second.last_access_time;
            lru_model_id = pair.first;
        }
    }

    if (!lru_model_id.empty()) {
        CRKIT_LOG_INFO("Evicting LRU model: ", lru_model_id);
        UnloadModel(lru_model_id);
    }
}

void ModelManager::UpdateAccessTime(const std::string& model_id) {
    auto it = models_.find(model_id);
    if (it != models_.end()) {
        it->second.last_access_time =
            std::chrono::system_clock::now().time_since_epoch().count();
    }
}

} // namespace crkit
