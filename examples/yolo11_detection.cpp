/**
 * YOLO11质检缺陷检测完整示例
 *
 * 本示例展示如何使用CRKIT SDK进行YOLO11模型推理
 * 适用于工业质检场景的缺陷检测任务
 */

#include <crkit/crkit.h>
#include <crkit/postprocess/yolo_postprocessor.h>
#include <iostream>
#include <vector>
#include <string>

// COCO类别名称（80类）
std::vector<std::string> GetCOCOClassNames() {
    return {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

// 工业质检类别名称（示例）
std::vector<std::string> GetDefectClassNames() {
    return {
        "scratch",      // 划痕
        "dent",         // 凹痕
        "crack",        // 裂纹
        "discoloration",// 变色
        "contamination",// 污染
        "burr",         // 毛刺
        "bubble",       // 气泡
        "deformation"   // 变形
    };
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "YOLO11 Defect Detection Example\n";
        std::cout << "Usage: " << argv[0] << " <model.onnx> <image.jpg> [conf_threshold] [iou_threshold]\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " yolo11n.onnx defect.jpg 0.5 0.45\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    float conf_threshold = argc > 3 ? std::atof(argv[3]) : 0.25f;
    float iou_threshold = argc > 4 ? std::atof(argv[4]) : 0.45f;

    std::cout << "=== CRKIT YOLO11 Defect Detection ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Confidence Threshold: " << conf_threshold << std::endl;
    std::cout << "NMS IOU Threshold: " << iou_threshold << std::endl;
    std::cout << std::endl;

    // 1. 初始化SDK
    crkit::Initialize();
    crkit::Logger::Instance().SetLogLevel(crkit::LogLevel::INFO);

    // 2. 配置YOLO后处理
    crkit::postprocess::YOLOConfig yolo_config;
    yolo_config.num_classes = 80;  // COCO数据集80类（根据实际模型调整）
    yolo_config.conf_threshold = conf_threshold;
    yolo_config.iou_threshold = iou_threshold;
    yolo_config.max_detections = 300;
    yolo_config.input_width = 640;
    yolo_config.input_height = 640;

    // 设置类别名称（根据实际训练的模型选择）
    // yolo_config.class_names = GetDefectClassNames();  // 工业质检类别
    yolo_config.class_names = GetCOCOClassNames();      // COCO类别

    // 3. 创建YOLO后处理器
    auto yolo_postprocessor = std::make_shared<crkit::postprocess::YOLOPostProcessor>(yolo_config);

    // 4. 配置推理引擎
    crkit::InferenceConfig config;
    config.engine_type = crkit::EngineType::ONNXRUNTIME;
    config.device_type = crkit::DeviceType::CPU;  // 或 GPU
    config.num_threads = 4;
    config.max_batch_size = 1;

    // 5. 创建推理引擎
    std::cout << "Loading model..." << std::endl;
    auto engine = crkit::CreateInferenceEngine(
        config.engine_type,
        model_path,
        config
    );

    if (!engine) {
        std::cerr << "Failed to create inference engine" << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "Engine: " << engine->GetEngineVersion() << std::endl;

    // 显示模型信息
    auto input_names = engine->GetInputNames();
    auto output_names = engine->GetOutputNames();
    std::cout << "\nModel Information:" << std::endl;
    std::cout << "  Inputs: " << input_names.size() << std::endl;
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto shape = engine->GetInputShape(input_names[i]);
        std::cout << "    - " << input_names[i] << ": [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "  Outputs: " << output_names.size() << std::endl;
    for (size_t i = 0; i < output_names.size(); ++i) {
        std::cout << "    - " << output_names[i] << std::endl;
    }

    // 6. 预热
    std::cout << "\nWarming up..." << std::endl;
    engine->WarmUp(5);
    std::cout << "Warm-up completed" << std::endl;

    // 7. 加载并预处理图像
    std::cout << "\nLoading image: " << image_path << std::endl;

    crkit::utils::ImageLoadOptions load_options;
    load_options.target_width = 640;
    load_options.target_height = 640;
    load_options.normalize = false;  // YOLO通常使用0-1归一化
    load_options.hwc_to_chw = true;  // YOLO需要CHW格式

    crkit::Tensor input_image = crkit::utils::LoadImage(image_path, load_options);

    if (input_image.Empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "Image loaded successfully" << std::endl;
    std::cout << "Input shape: [";
    const auto& shape = input_image.GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 添加batch维度 [C, H, W] -> [1, C, H, W]
    crkit::Tensor batch_input({1, shape[0], shape[1], shape[2]}, crkit::DataType::FLOAT32);
    std::memcpy(batch_input.Data(), input_image.Data(), input_image.ByteSize());

    // 8. 执行推理
    std::cout << "\nRunning inference..." << std::endl;
    crkit::InferenceResult raw_result;
    auto status = engine->Infer(batch_input, raw_result);

    if (!status.IsOK()) {
        std::cerr << "Inference failed: " << status.Message() << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "Inference completed in " << raw_result.GetInferenceTime() << " ms" << std::endl;
    std::cout << "Raw outputs: " << raw_result.raw_outputs.size() << std::endl;

    // 9. 后处理（YOLO解析）
    std::cout << "\nPost-processing..." << std::endl;
    crkit::InferenceResult final_result;
    status = yolo_postprocessor->Process(raw_result.raw_outputs, final_result);

    if (!status.IsOK()) {
        std::cerr << "Post-processing failed: " << status.Message() << std::endl;
        crkit::Shutdown();
        return 1;
    }

    // 保留推理时间
    final_result.SetInferenceTime(raw_result.GetInferenceTime());

    // 10. 输出检测结果
    std::cout << "\n=== Detection Results ===" << std::endl;
    std::cout << "Total detections: " << final_result.detections.size() << std::endl;
    std::cout << "Inference time: " << final_result.GetInferenceTime() << " ms" << std::endl;
    std::cout << "FPS: " << (1000.0f / final_result.GetInferenceTime()) << std::endl;

    if (final_result.detections.empty()) {
        std::cout << "\nNo defects detected!" << std::endl;
    } else {
        std::cout << "\nDetected defects:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::setw(5) << "ID"
                  << std::setw(20) << "Class"
                  << std::setw(12) << "Confidence"
                  << std::setw(15) << "X"
                  << std::setw(10) << "Y"
                  << std::setw(10) << "Width"
                  << std::setw(10) << "Height" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (size_t i = 0; i < final_result.detections.size(); ++i) {
            const auto& det = final_result.detections[i];
            std::cout << std::setw(5) << (i + 1)
                      << std::setw(20) << det.label
                      << std::setw(12) << std::fixed << std::setprecision(3) << det.confidence
                      << std::setw(15) << std::fixed << std::setprecision(1) << det.bbox.x
                      << std::setw(10) << det.bbox.y
                      << std::setw(10) << det.bbox.width
                      << std::setw(10) << det.bbox.height << std::endl;
        }
        std::cout << std::string(80, '-') << std::endl;

        // 11. 可视化结果（保存带检测框的图像）
        std::cout << "\nSaving detection result..." << std::endl;

        // 加载原始图像用于可视化（HWC格式）
        crkit::utils::ImageLoadOptions viz_options;
        viz_options.target_width = 640;
        viz_options.target_height = 640;
        viz_options.normalize = false;
        viz_options.hwc_to_chw = false;  // 保持HWC格式用于OpenCV

        crkit::Tensor viz_image = crkit::utils::LoadImage(image_path, viz_options);

        // 绘制检测框
        crkit::utils::ImageUtils::DrawDetections(viz_image, final_result.detections, 2);

        // 保存结果
        std::string output_path = "detection_result.jpg";
        auto save_status = crkit::utils::ImageUtils::SaveImage(output_path, viz_image);

        if (save_status.IsOK()) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save result image: " << save_status.Message() << std::endl;
        }

        // 统计各类别数量
        std::map<std::string, int> class_counts;
        for (const auto& det : final_result.detections) {
            class_counts[det.label]++;
        }

        std::cout << "\nDetections by class:" << std::endl;
        for (const auto& [label, count] : class_counts) {
            std::cout << "  " << label << ": " << count << std::endl;
        }
    }

    // 12. 清理资源
    engine->Release();
    crkit::Shutdown();

    std::cout << "\nDone!" << std::endl;
    return 0;
}
