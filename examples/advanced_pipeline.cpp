/**
 * 高级Pipeline示例
 * 展示如何使用预处理/后处理Pipeline和模型管理器
 */

#include <crkit/crkit.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // 初始化SDK
    crkit::Initialize();

    // 配置日志
    crkit::Logger::Instance().SetLogLevel(crkit::LogLevel::INFO);

    // 注册模型到模型管理器
    crkit::ModelInfo model_info;
    model_info.model_id = "defect_detector";
    model_info.model_path = model_path;
    model_info.engine_type = crkit::EngineType::TENSORRT;
    model_info.description = "工业质检缺陷检测模型";
    model_info.config.device_type = crkit::DeviceType::GPU;
    model_info.config.enable_fp16 = true;

    auto& model_mgr = crkit::ModelManager::Instance();
    auto status = model_mgr.RegisterModel(model_info);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Failed to register model: ", status.Message());
        return 1;
    }

    // 加载模型
    status = model_mgr.LoadModel("defect_detector");
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Failed to load model: ", status.Message());
        return 1;
    }

    // 构建预处理Pipeline
    auto preprocess = crkit::PreProcessPipelineBuilder()
        .Resize(640, 640, true)              // 调整到640x640，保持宽高比
        .ColorConvert(                        // BGR转RGB
            crkit::preprocessing::ColorConvert::ColorSpace::BGR,
            crkit::preprocessing::ColorConvert::ColorSpace::RGB)
        .NormalizeImageNet()                  // ImageNet归一化
        .HWCToCHW()                           // HWC转CHW
        .Build();

    // 构建后处理Pipeline
    auto postprocess = crkit::PostProcessPipelineBuilder()
        .FilterByConfidence(0.5f)             // 过滤置信度<0.5的结果
        .ApplyNMS(0.45f)                      // NMS，IOU阈值0.45
        .SelectTopK(100)                      // 只保留前100个检测
        .Build();

    // 创建推理上下文
    auto context = crkit::InferenceContextBuilder()
        .WithEngine(model_mgr.GetEngine("defect_detector"))
        .Build();

    // 加载图像
    crkit::Tensor raw_input = crkit::utils::LoadImage(image_path);
    if (raw_input.Empty()) {
        CRKIT_LOG_ERROR("Failed to load image: ", image_path);
        return 1;
    }

    CRKIT_LOG_INFO("Image loaded successfully");

    // 预处理
    crkit::Tensor preprocessed_input;
    status = preprocess->Execute(raw_input, preprocessed_input);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Preprocessing failed: ", status.Message());
        return 1;
    }

    CRKIT_LOG_INFO("Preprocessing completed");

    // 推理
    crkit::InferenceResult raw_result;
    status = context->InferRaw(preprocessed_input, raw_result);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Inference failed: ", status.Message());
        return 1;
    }

    CRKIT_LOG_INFO("Inference completed in ", raw_result.GetInferenceTime(), " ms");

    // 后处理
    crkit::InferenceResult final_result;
    status = postprocess->Execute(raw_result, final_result);
    if (!status.IsOK()) {
        CRKIT_LOG_ERROR("Postprocessing failed: ", status.Message());
        return 1;
    }

    // 输出结果统计
    std::cout << "\n=== Detection Results ===" << std::endl;
    std::cout << "Total detections: " << final_result.detections.size() << std::endl;
    std::cout << "Inference time: " << final_result.GetInferenceTime() << " ms" << std::endl;

    // 统计各类别检测数量
    std::map<std::string, int> class_counts;
    for (const auto& det : final_result.detections) {
        class_counts[det.label]++;
    }

    std::cout << "\nDetections by class:" << std::endl;
    for (const auto& [label, count] : class_counts) {
        std::cout << "  " << label << ": " << count << std::endl;
    }

    // 显示高置信度检测
    std::cout << "\nHigh confidence detections (>0.8):" << std::endl;
    for (const auto& det : final_result.detections) {
        if (det.confidence > 0.8f) {
            std::cout << "  " << det.label << " - " << det.confidence
                     << " @ [" << det.bbox.x << ", " << det.bbox.y << "]"
                     << std::endl;
        }
    }

    // 可视化结果（如果需要）
    // crkit::Tensor result_image = raw_input.Clone();
    // crkit::utils::ImageUtils::DrawDetections(result_image, final_result.detections);
    // crkit::utils::ImageUtils::SaveImage("result.jpg", result_image);

    // 输出模型统计信息
    auto model_stats = model_mgr.GetModelStats("defect_detector");
    std::cout << "\n=== Model Statistics ===" << std::endl;
    std::cout << "Total inferences: " << model_stats.inference_count << std::endl;
    std::cout << "Average time: " << model_stats.avg_time_ms << " ms" << std::endl;

    // 清理
    model_mgr.UnloadModel("defect_detector");
    crkit::Shutdown();

    return 0;
}
