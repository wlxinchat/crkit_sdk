#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace crkit {

// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

// 日志器
class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    void SetLogLevel(LogLevel level) { log_level_ = level; }
    LogLevel GetLogLevel() const { return log_level_; }

    void SetEnableConsole(bool enable) { enable_console_ = enable; }
    void SetEnableFile(bool enable) { enable_file_ = enable; }
    void SetLogFile(const std::string& filepath) { log_file_ = filepath; }

    template<typename... Args>
    void Log(LogLevel level, const char* file, int line, Args&&... args) {
        if (level < log_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream oss;
        oss << "[" << GetTimestamp() << "] "
            << "[" << LogLevelToString(level) << "] "
            << "[" << file << ":" << line << "] ";

        LogImpl(oss, std::forward<Args>(args)...);

        std::string message = oss.str();

        if (enable_console_) {
            std::cout << message << std::endl;
        }

        if (enable_file_ && !log_file_.empty()) {
            // TODO: 写入文件
        }
    }

private:
    Logger() : log_level_(LogLevel::INFO),
               enable_console_(true),
               enable_file_(false) {}

    template<typename T>
    void LogImpl(std::ostringstream& oss, T&& arg) {
        oss << std::forward<T>(arg);
    }

    template<typename T, typename... Args>
    void LogImpl(std::ostringstream& oss, T&& arg, Args&&... args) {
        oss << std::forward<T>(arg);
        LogImpl(oss, std::forward<Args>(args)...);
    }

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

    std::string LogLevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::FATAL:   return "FATAL";
            default:                return "UNKNOWN";
        }
    }

    LogLevel log_level_;
    bool enable_console_;
    bool enable_file_;
    std::string log_file_;
    std::mutex mutex_;
};

} // namespace crkit

// 日志宏
#define CRKIT_LOG_DEBUG(...)   crkit::Logger::Instance().Log(crkit::LogLevel::DEBUG,   __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_INFO(...)    crkit::Logger::Instance().Log(crkit::LogLevel::INFO,    __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_WARNING(...) crkit::Logger::Instance().Log(crkit::LogLevel::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_ERROR(...)   crkit::Logger::Instance().Log(crkit::LogLevel::ERROR,   __FILE__, __LINE__, __VA_ARGS__)
#define CRKIT_LOG_FATAL(...)   crkit::Logger::Instance().Log(crkit::LogLevel::FATAL,   __FILE__, __LINE__, __VA_ARGS__)
