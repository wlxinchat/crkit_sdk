#pragma once

// CRKIT SDK - 工业质检AI推理SDK主头文件
// 包含此文件即可使用SDK的所有功能

// 核心类型
#include "crkit/core/types.h"

// 数据结构
#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"

// 推理引擎
#include "crkit/engine/inference_engine.h"

// 核心功能
#include "crkit/core/inference_context.h"
#include "crkit/core/model_manager.h"

// 工具类
#include "crkit/utils/logger.h"
#include "crkit/utils/image_utils.h"
#include "crkit/utils/pipeline.h"

// 版本信息
#define CRKIT_VERSION_MAJOR 1
#define CRKIT_VERSION_MINOR 0
#define CRKIT_VERSION_PATCH 0
#define CRKIT_VERSION "1.0.0"

namespace crkit {

// 初始化SDK
inline Status Initialize() {
    CRKIT_LOG_INFO("CRKIT SDK v", CRKIT_VERSION, " initialized");
    return Status();
}

// 清理SDK资源
inline void Shutdown() {
    ModelManager::Instance().UnloadAllModels();
    CRKIT_LOG_INFO("CRKIT SDK shutdown");
}

// 获取版本信息
inline std::string GetVersion() {
    return CRKIT_VERSION;
}

} // namespace crkit
