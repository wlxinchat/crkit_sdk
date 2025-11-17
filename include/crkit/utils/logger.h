#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cstring>
#include <filesystem>
#include <map>

namespace crkit {

// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

// 日志模块
enum class LogModule {
    CORE,       // 核心功能
    ENGINE,     // 推理引擎
    ONNX,       // ONNX Runtime
    TENSORRT,   // TensorRT
    OPENVINO,   // OpenVINO
    POSTPROC,   // 后处理
    UTILS,      // 工具类
    IMAGE,      // 图像处理
    MODEL,      // 模型管理
    ALL         // 所有模块（用于配置）
};

// 日志器配置
struct LoggerConfig {
    LogLevel global_level = LogLevel::INFO;
    bool enable_console = true;
    bool enable_file = false;
    bool enable_color = true;      // 控制台颜色
    std::string log_file = "logs/crkit.log";
    size_t max_file_size = 100 * 1024 * 1024;  // 100MB
    int max_file_count = 5;
    bool auto_flush = true;        // 自动刷新（可靠性优先）
};

// 日志器
class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    // 配置接口
    void Configure(const LoggerConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        if (config_.enable_file) {
            OpenLogFile();
        }
    }

    void SetLogLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.global_level = level;
    }

    void SetModuleLogLevel(LogModule module, LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        module_levels_[module] = level;
    }

    LogLevel GetLogLevel() const { return config_.global_level; }

    void SetEnableConsole(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.enable_console = enable;
    }

    void SetEnableFile(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.enable_file = enable;
        if (enable) {
            OpenLogFile();
        } else {
            CloseLogFile();
        }
    }

    void SetEnableColor(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.enable_color = enable;
    }

    void SetLogFile(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_.log_file = filepath;
        if (config_.enable_file) {
            CloseLogFile();
            OpenLogFile();
        }
    }

    // 主日志接口
    template<typename... Args>
    void Log(LogModule module, LogLevel level, const char* file, int line, Args&&... args) {
        // 检查日志级别
        if (!ShouldLog(module, level)) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // 构建日志消息
        std::ostringstream oss;

        // 时间戳
        oss << "[" << GetTimestamp() << "] ";

        // 日志级别（带颜色）
        if (config_.enable_console && config_.enable_color) {
            oss << GetColorCode(level);
        }
        oss << "[" << LogLevelToString(level) << "]";
        if (config_.enable_console && config_.enable_color) {
            oss << RESET;
        }

        // 模块标识
        oss << " [" << LogModuleToString(module) << "]";

        // 线程ID
        oss << " [T:" << GetThreadId() << "]";

        // 文件名和行号
        oss << " [" << GetBasename(file) << ":" << line << "] ";

        // 日志内容
        LogImpl(oss, std::forward<Args>(args)...);

        std::string message = oss.str();

        // 输出到控制台
        if (config_.enable_console) {
            std::cout << message << std::endl;
        }

        // 输出到文件
        if (config_.enable_file) {
            WriteToFile(message);
        }
    }

    // 刷新日志文件
    void Flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (log_file_stream_.is_open()) {
            log_file_stream_.flush();
        }
    }

private:
    Logger() = default;

    ~Logger() {
        CloseLogFile();
    }

    // 禁止拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // 检查是否应该记录日志
    bool ShouldLog(LogModule module, LogLevel level) {
        // 检查模块级别
        auto it = module_levels_.find(module);
        if (it != module_levels_.end()) {
            return level >= it->second;
        }
        // 使用全局级别
        return level >= config_.global_level;
    }

    // 递归展开参数
    template<typename T>
    void LogImpl(std::ostringstream& oss, T&& arg) {
        oss << std::forward<T>(arg);
    }

    template<typename T, typename... Args>
    void LogImpl(std::ostringstream& oss, T&& arg, Args&&... args) {
        oss << std::forward<T>(arg);
        LogImpl(oss, std::forward<Args>(args)...);
    }

    // 获取时间戳
    std::string GetTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    // 获取线程ID
    std::string GetThreadId() {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        return oss.str();
    }

    // 获取文件名（去除路径）
    static const char* GetBasename(const char* filepath) {
        const char* base = std::strrchr(filepath, '/');
        if (!base) base = std::strrchr(filepath, '\\');
        return base ? base + 1 : filepath;
    }

    // 日志级别转字符串
    std::string LogLevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO ";
            case LogLevel::WARNING: return "WARN ";
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::FATAL:   return "FATAL";
            default:                return "UNKNO";
        }
    }

    // 模块转字符串
    std::string LogModuleToString(LogModule module) {
        switch (module) {
            case LogModule::CORE:      return "CORE    ";
            case LogModule::ENGINE:    return "ENGINE  ";
            case LogModule::ONNX:      return "ONNX    ";
            case LogModule::TENSORRT:  return "TENSORRT";
            case LogModule::OPENVINO:  return "OPENVINO";
            case LogModule::POSTPROC:  return "POSTPROC";
            case LogModule::UTILS:     return "UTILS   ";
            case LogModule::IMAGE:     return "IMAGE   ";
            case LogModule::MODEL:     return "MODEL   ";
            case LogModule::ALL:       return "ALL     ";
            default:                   return "UNKNOWN ";
        }
    }

    // ANSI颜色代码
    const char* GetColorCode(LogLevel level) {
        if (!config_.enable_color) return "";
        switch (level) {
            case LogLevel::DEBUG:   return "\033[36m";  // 青色
            case LogLevel::INFO:    return "\033[32m";  // 绿色
            case LogLevel::WARNING: return "\033[33m";  // 黄色
            case LogLevel::ERROR:   return "\033[31m";  // 红色
            case LogLevel::FATAL:   return "\033[35m";  // 紫色
            default:                return "\033[0m";   // 默认
        }
    }
    static constexpr const char* RESET = "\033[0m";

    // 打开日志文件
    void OpenLogFile() {
        // 确保目录存在
        std::filesystem::path log_path(config_.log_file);
        std::filesystem::path log_dir = log_path.parent_path();
        if (!log_dir.empty() && !std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }

        // 检查是否需要轮转
        if (std::filesystem::exists(config_.log_file)) {
            auto file_size = std::filesystem::file_size(config_.log_file);
            if (file_size >= config_.max_file_size) {
                RotateLogFile();
            }
        }

        // 打开文件（追加模式）
        log_file_stream_.open(config_.log_file, std::ios::app);
        if (!log_file_stream_.is_open()) {
            std::cerr << "Failed to open log file: " << config_.log_file << std::endl;
        }
    }

    // 关闭日志文件
    void CloseLogFile() {
        if (log_file_stream_.is_open()) {
            log_file_stream_.close();
        }
    }

    // 日志文件轮转
    void RotateLogFile() {
        if (!std::filesystem::exists(config_.log_file)) return;

        // 关闭当前文件
        CloseLogFile();

        // 删除最老的日志文件
        std::string oldest_log = config_.log_file + "." + std::to_string(config_.max_file_count);
        if (std::filesystem::exists(oldest_log)) {
            std::filesystem::remove(oldest_log);
        }

        // 轮转：log.4 <- log.3 <- log.2 <- log.1 <- log
        for (int i = config_.max_file_count - 1; i >= 1; --i) {
            std::string old_name = config_.log_file + "." + std::to_string(i);
            std::string new_name = config_.log_file + "." + std::to_string(i + 1);
            if (std::filesystem::exists(old_name)) {
                std::filesystem::rename(old_name, new_name);
            }
        }

        // 重命名当前日志文件
        std::string backup_name = config_.log_file + ".1";
        std::filesystem::rename(config_.log_file, backup_name);
    }

    // 写入文件
    void WriteToFile(const std::string& message) {
        if (!log_file_stream_.is_open()) {
            OpenLogFile();
        }

        if (log_file_stream_.is_open()) {
            // 写入时去除ANSI颜色代码（文件中不需要颜色）
            std::string clean_message = RemoveColorCodes(message);
            log_file_stream_ << clean_message << std::endl;

            if (config_.auto_flush) {
                log_file_stream_.flush();
            }

            // 检查文件大小，需要时轮转
            if (log_file_stream_.tellp() >= static_cast<std::streampos>(config_.max_file_size)) {
                RotateLogFile();
            }
        }
    }

    // 移除ANSI颜色代码
    std::string RemoveColorCodes(const std::string& str) {
        std::string result;
        result.reserve(str.size());
        bool in_escape = false;

        for (char c : str) {
            if (c == '\033') {
                in_escape = true;
            } else if (in_escape && c == 'm') {
                in_escape = false;
            } else if (!in_escape) {
                result += c;
            }
        }

        return result;
    }

    LoggerConfig config_;
    std::map<LogModule, LogLevel> module_levels_;
    std::ofstream log_file_stream_;
    std::mutex mutex_;
};

} // namespace crkit

// 日志宏 - 带模块标识（推荐使用）
#define CRKIT_LOG_DEBUG_M(module, ...)   crkit::Logger::Instance().Log(crkit::LogModule::module, crkit::LogLevel::DEBUG,   __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_INFO_M(module, ...)    crkit::Logger::Instance().Log(crkit::LogModule::module, crkit::LogLevel::INFO,    __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_WARNING_M(module, ...) crkit::Logger::Instance().Log(crkit::LogModule::module, crkit::LogLevel::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_ERROR_M(module, ...)   crkit::Logger::Instance().Log(crkit::LogModule::module, crkit::LogLevel::ERROR,   __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_FATAL_M(module, ...)   crkit::Logger::Instance().Log(crkit::LogModule::module, crkit::LogLevel::FATAL,   __FILE__, __LINE__, __VA_ARGS__)

// 日志宏 - 兼容旧版本（默认使用CORE模块）
#define CRKIT_LOG_DEBUG(...)   CRKIT_LOG_DEBUG_M(CORE, __VA_ARGS__)
#define CRKIT_LOG_INFO(...)    CRKIT_LOG_INFO_M(CORE, __VA_ARGS__)
#define CRKIT_LOG_WARNING(...) CRKIT_LOG_WARNING_M(CORE, __VA_ARGS__)
#define CRKIT_LOG_ERROR(...)   CRKIT_LOG_ERROR_M(CORE, __VA_ARGS__)
#define CRKIT_LOG_FATAL(...)   CRKIT_LOG_FATAL_M(CORE, __VA_ARGS__)
