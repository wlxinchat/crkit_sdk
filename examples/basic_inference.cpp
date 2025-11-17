/**
 * 基础推理示例
 * 展示如何使用CRKIT SDK进行简单的图像推理
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

    // 配置推理引擎
    crkit::InferenceConfig config;
    config.engine_type = crkit::EngineType::TENSORRT;  // 使用TensorRT
    config.device_type = crkit::DeviceType::GPU;       // GPU推理
    config.device_id = 0;                               // GPU 0
    config.max_batch_size = 1;
    config.enable_fp16 = true;                          // 启用FP16加速

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

    // 预热推理引擎
    std::cout << "Warming up inference engine..." << std::endl;
    engine->WarmUp(10);

    // 加载图像
    crkit::utils::ImageLoadOptions load_options;
    load_options.target_width = 640;
    load_options.target_height = 640;
    load_options.normalize = true;
    load_options.mean = {0.485f, 0.456f, 0.406f};
    load_options.std = {0.229f, 0.224f, 0.225f};
    load_options.hwc_to_chw = true;

    crkit::Tensor input = crkit::utils::LoadImage(image_path, load_options);
    if (input.Empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << image_path << std::endl;
    std::cout << "Input shape: [";
    for (size_t i = 0; i < input.GetShape().size(); ++i) {
        std::cout << input.GetShape()[i];
        if (i < input.GetShape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 执行推理
    crkit::InferenceResult result;
    auto status = engine->Infer(input, result);

    if (!status.IsOK()) {
        std::cerr << "Inference failed: " << status.Message() << std::endl;
        return 1;
    }

    // 输出结果
    std::cout << "\n=== Inference Results ===" << std::endl;
    std::cout << "Inference time: " << result.GetInferenceTime() << " ms" << std::endl;
    std::cout << "Detections: " << result.detections.size() << std::endl;

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        std::cout << "\nDetection " << i + 1 << ":" << std::endl;
        std::cout << "  Label: " << det.label << std::endl;
        std::cout << "  Confidence: " << det.confidence << std::endl;
        std::cout << "  BBox: [" << det.bbox.x << ", " << det.bbox.y << ", "
                  << det.bbox.width << ", " << det.bbox.height << "]" << std::endl;
    }

    // 清理资源
    engine->Release();
    crkit::Shutdown();

    return 0;
}
