/**
 * 日志系统测试程序
 * 测试模块标识、线程ID、文件日志、日志轮转等功能
 */

#include "crkit/utils/logger.h"
#include <thread>
#include <vector>
#include <chrono>

using namespace crkit;

// 测试基础日志功能
void test_basic_logging() {
    std::cout << "\n=== 测试1: 基础日志功能 ===\n" << std::endl;

    CRKIT_LOG_DEBUG("This is a DEBUG message");
    CRKIT_LOG_INFO("This is an INFO message");
    CRKIT_LOG_WARNING("This is a WARNING message");
    CRKIT_LOG_ERROR("This is an ERROR message");
    CRKIT_LOG_FATAL("This is a FATAL message");
}

// 测试模块标识
void test_module_logging() {
    std::cout << "\n=== 测试2: 模块标识 ===\n" << std::endl;

    CRKIT_LOG_INFO_M(CORE, "Core module message");
    CRKIT_LOG_INFO_M(ENGINE, "Engine module message");
    CRKIT_LOG_INFO_M(ONNX, "ONNX module message");
    CRKIT_LOG_INFO_M(POSTPROC, "Post-process module message");
    CRKIT_LOG_INFO_M(UTILS, "Utils module message");
    CRKIT_LOG_INFO_M(MODEL, "Model module message");
}

// 测试多线程日志
void thread_worker(int thread_id) {
    for (int i = 0; i < 3; ++i) {
        CRKIT_LOG_INFO_M(ENGINE, "Thread ", thread_id, " iteration ", i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void test_multithreaded_logging() {
    std::cout << "\n=== 测试3: 多线程日志 ===\n" << std::endl;

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(thread_worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }
}

// 测试日志级别过滤
void test_log_level_filtering() {
    std::cout << "\n=== 测试4: 日志级别过滤 ===\n" << std::endl;

    std::cout << "设置级别为WARNING，DEBUG和INFO应该被过滤" << std::endl;
    Logger::Instance().SetLogLevel(LogLevel::WARNING);

    CRKIT_LOG_DEBUG("This DEBUG should NOT appear");
    CRKIT_LOG_INFO("This INFO should NOT appear");
    CRKIT_LOG_WARNING("This WARNING should appear");
    CRKIT_LOG_ERROR("This ERROR should appear");

    // 恢复INFO级别
    Logger::Instance().SetLogLevel(LogLevel::INFO);
    std::cout << "恢复级别为INFO" << std::endl;
}

// 测试模块级日志控制
void test_module_level_control() {
    std::cout << "\n=== 测试5: 模块级日志控制 ===\n" << std::endl;

    std::cout << "设置ENGINE模块为ERROR级别" << std::endl;
    Logger::Instance().SetModuleLogLevel(LogModule::ENGINE, LogLevel::ERROR);

    CRKIT_LOG_INFO_M(ENGINE, "ENGINE INFO should NOT appear");
    CRKIT_LOG_ERROR_M(ENGINE, "ENGINE ERROR should appear");
    CRKIT_LOG_INFO_M(CORE, "CORE INFO should appear");
}

// 测试文件日志
void test_file_logging() {
    std::cout << "\n=== 测试6: 文件日志 ===\n" << std::endl;

    LoggerConfig config;
    config.enable_file = true;
    config.enable_console = true;
    config.log_file = "logs/test_logger.log";
    config.max_file_size = 1024;  // 1KB for testing rotation
    config.max_file_count = 3;

    Logger::Instance().Configure(config);

    std::cout << "开启文件日志: " << config.log_file << std::endl;

    for (int i = 0; i < 10; ++i) {
        CRKIT_LOG_INFO_M(UTILS, "File logging test message ", i, 
                         " - This is a longer message to test file rotation");
    }

    Logger::Instance().Flush();
    std::cout << "日志已写入文件（可能触发轮转）" << std::endl;
}

// 测试颜色输出
void test_color_output() {
    std::cout << "\n=== 测试7: 颜色输出 ===\n" << std::endl;

    std::cout << "彩色输出已启用，不同级别应显示不同颜色" << std::endl;
    Logger::Instance().SetEnableColor(true);

    CRKIT_LOG_DEBUG("DEBUG - 应该是青色");
    CRKIT_LOG_INFO("INFO - 应该是绿色");
    CRKIT_LOG_WARNING("WARNING - 应该是黄色");
    CRKIT_LOG_ERROR("ERROR - 应该是红色");
    CRKIT_LOG_FATAL("FATAL - 应该是紫色");
}

// 测试格式化输出
void test_formatted_output() {
    std::cout << "\n=== 测试8: 格式化输出 ===\n" << std::endl;

    int value = 42;
    float pi = 3.14159f;
    std::string name = "CRKIT SDK";

    CRKIT_LOG_INFO_M(CORE, "Integer: ", value, ", Float: ", pi, ", String: ", name);
    CRKIT_LOG_INFO_M(ENGINE, "Formatted: value=", value, " pi=", pi);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CRKIT SDK 日志系统测试" << std::endl;
    std::cout << "========================================" << std::endl;

    // 配置日志系统
    LoggerConfig config;
    config.global_level = LogLevel::DEBUG;
    config.enable_console = true;
    config.enable_color = true;
    Logger::Instance().Configure(config);

    // 运行测试
    test_basic_logging();
    test_module_logging();
    test_multithreaded_logging();
    test_log_level_filtering();
    test_module_level_control();
    test_file_logging();
    test_color_output();
    test_formatted_output();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  测试完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
