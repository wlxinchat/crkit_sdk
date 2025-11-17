/**
 * 批量推理示例
 * 展示如何进行批量图像推理，适用于工业质检场景
 */

#include <crkit/crkit.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_directory>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_dir = argv[2];

    // 初始化
    crkit::Initialize();

    // 收集所有图像文件
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
    }

    if (image_files.empty()) {
        std::cerr << "No images found in directory: " << image_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << image_files.size() << " images" << std::endl;

    // 配置推理引擎
    crkit::InferenceConfig config;
    config.engine_type = crkit::EngineType::TENSORRT;
    config.device_type = crkit::DeviceType::GPU;
    config.max_batch_size = 8;  // 批量大小为8
    config.enable_fp16 = true;

    // 创建推理引擎
    auto engine = crkit::CreateInferenceEngine(
        crkit::EngineType::TENSORRT,
        model_path,
        config
    );

    if (!engine) {
        std::cerr << "Failed to create inference engine" << std::endl;
        return 1;
    }

    // 预热
    std::cout << "Warming up..." << std::endl;
    engine->WarmUp(5);

    // 配置图像加载选项
    crkit::utils::ImageLoadOptions load_options;
    load_options.target_width = 640;
    load_options.target_height = 640;
    load_options.normalize = true;
    load_options.mean = {0.485f, 0.456f, 0.406f};
    load_options.std = {0.229f, 0.224f, 0.225f};
    load_options.hwc_to_chw = true;

    // 批量推理统计
    int total_defects = 0;
    float total_inference_time = 0.0f;
    const int batch_size = config.max_batch_size;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 分批处理图像
    for (size_t i = 0; i < image_files.size(); i += batch_size) {
        // 准备当前批次
        size_t current_batch_size = std::min(batch_size,
                                             static_cast<int>(image_files.size() - i));

        std::vector<crkit::Tensor> batch_inputs;
        std::vector<std::string> batch_files;

        // 加载批次图像
        for (size_t j = 0; j < current_batch_size; ++j) {
            std::string filepath = image_files[i + j];
            crkit::Tensor input = crkit::utils::LoadImage(filepath, load_options);

            if (!input.Empty()) {
                batch_inputs.push_back(input);
                batch_files.push_back(filepath);
            } else {
                std::cerr << "Warning: Failed to load " << filepath << std::endl;
            }
        }

        if (batch_inputs.empty()) {
            continue;
        }

        // 批量推理
        std::vector<crkit::InferenceResult> results;
        auto status = engine->InferBatch(batch_inputs, results);

        if (!status.IsOK()) {
            std::cerr << "Batch inference failed: " << status.Message() << std::endl;
            continue;
        }

        // 处理结果
        for (size_t j = 0; j < results.size(); ++j) {
            const auto& result = results[j];
            const auto& filepath = batch_files[j];

            total_inference_time += result.GetInferenceTime();

            // 过滤低置信度检测
            auto filtered_detections = result.detections;
            filtered_detections.erase(
                std::remove_if(filtered_detections.begin(), filtered_detections.end(),
                              [](const crkit::Detection& det) {
                                  return det.confidence < 0.5f;
                              }),
                filtered_detections.end()
            );

            int defect_count = filtered_detections.size();
            total_defects += defect_count;

            // 输出每张图片的结果
            std::cout << "[" << (i + j + 1) << "/" << image_files.size() << "] "
                     << fs::path(filepath).filename().string()
                     << " - Defects: " << defect_count
                     << " (" << result.GetInferenceTime() << " ms)"
                     << std::endl;

            // 如果发现缺陷，显示详细信息
            if (defect_count > 0) {
                for (const auto& det : filtered_detections) {
                    std::cout << "    └─ " << det.label
                             << " (conf: " << det.confidence << ")"
                             << std::endl;
                }
            }
        }

        // 显示批次处理进度
        float progress = static_cast<float>(std::min(i + batch_size, image_files.size()))
                        / image_files.size() * 100.0f;
        std::cout << "Progress: " << std::fixed << std::setprecision(1)
                 << progress << "%" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // 输出统计信息
    std::cout << "\n=== Processing Summary ===" << std::endl;
    std::cout << "Total images processed: " << image_files.size() << std::endl;
    std::cout << "Total defects found: " << total_defects << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Average time per image: "
             << static_cast<float>(duration) / image_files.size() << " ms" << std::endl;
    std::cout << "Average inference time: "
             << total_inference_time / image_files.size() << " ms" << std::endl;
    std::cout << "Throughput: "
             << static_cast<float>(image_files.size()) / duration * 1000.0f
             << " images/sec" << std::endl;

    // 计算缺陷率
    float defect_rate = static_cast<float>(total_defects) / image_files.size();
    std::cout << "Average defects per image: " << defect_rate << std::endl;

    // 清理
    engine->Release();
    crkit::Shutdown();

    return 0;
}
