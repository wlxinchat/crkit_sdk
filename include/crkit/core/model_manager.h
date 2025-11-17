#pragma once

#include "crkit/engine/inference_engine.h"
#include "crkit/core/inference_context.h"
#include <memory>
#include <string>
#include <map>
#include <mutex>

namespace crkit {

// 模型信息
struct ModelInfo {
    std::string model_id;                    // 模型唯一标识
    std::string model_path;                  // 模型文件路径
    EngineType engine_type;                  // 推理引擎类型
    InferenceConfig config;                  // 推理配置
    std::string description;                 // 模型描述
    std::map<std::string, std::string> metadata; // 元数据

    ModelInfo() : engine_type(EngineType::AUTO) {}
};

// 模型管理器 - 管理多个模型的加载、卸载、切换
class ModelManager {
public:
    static ModelManager& Instance();

    // 注册模型（不立即加载）
    Status RegisterModel(const ModelInfo& model_info);

    // 加载模型
    Status LoadModel(const std::string& model_id);

    // 卸载模型
    Status UnloadModel(const std::string& model_id);

    // 获取推理引擎
    std::shared_ptr<IInferenceEngine> GetEngine(const std::string& model_id);

    // 获取推理上下文
    std::shared_ptr<InferenceContext> GetContext(const std::string& model_id);

    // 模型是否已加载
    bool IsModelLoaded(const std::string& model_id) const;

    // 获取模型信息
    ModelInfo GetModelInfo(const std::string& model_id) const;

    // 列出所有已注册的模型
    std::vector<std::string> ListRegisteredModels() const;

    // 列出所有已加载的模型
    std::vector<std::string> ListLoadedModels() const;

    // 热更新模型（不影响正在进行的推理）
    Status HotSwapModel(const std::string& model_id,
                       const std::string& new_model_path);

    // 批量加载模型
    Status LoadModels(const std::vector<std::string>& model_ids);

    // 卸载所有模型
    void UnloadAllModels();

    // 清理未使用的模型（基于最后使用时间）
    void CleanupUnusedModels(int64_t idle_time_seconds);

    // 获取模型使用统计
    struct ModelStats {
        std::string model_id;
        size_t inference_count;
        float total_time_ms;
        float avg_time_ms;
        int64_t last_used_timestamp;
        size_t memory_usage_bytes;
    };
    ModelStats GetModelStats(const std::string& model_id) const;

    // 设置最大同时加载的模型数量
    void SetMaxLoadedModels(size_t max_models);

private:
    ModelManager() : max_loaded_models_(10) {}
    ~ModelManager() { UnloadAllModels(); }

    // 禁止拷贝
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    struct ModelEntry {
        ModelInfo info;
        std::shared_ptr<IInferenceEngine> engine;
        std::shared_ptr<InferenceContext> context;
        ModelStats stats;
        int64_t last_access_time;
        bool is_loaded;
    };

    std::map<std::string, ModelEntry> models_;
    mutable std::mutex mutex_;
    size_t max_loaded_models_;

    // 自动卸载最久未使用的模型
    void EvictLRUModel();
    void UpdateAccessTime(const std::string& model_id);
};

} // namespace crkit
