/**
 * 通用目标检测示例
 *
 * 支持各种目标检测模型（如YOLO系列、SSD、RetinaNet等）
 * 只需配置正确的输出格式和后处理参数
 */

#include <crkit/crkit.h>
#include <crkit/postprocess/detection_postprocessor.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

// COCO数据集80类类别名称
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "通用目标检测示例\n";
        std::cout << "用法: " << argv[0] << " <model.onnx> <image.jpg> [conf_threshold] [iou_threshold]\n";
        std::cout << "\n参数说明:\n";
        std::cout << "  model.onnx      - ONNX模型路径\n";
        std::cout << "  image.jpg       - 输入图像路径\n";
        std::cout << "  conf_threshold  - 置信度阈值 (默认: 0.25)\n";
        std::cout << "  iou_threshold   - NMS IOU阈值 (默认: 0.45)\n";
        std::cout << "\n支持的模型:\n";
        std::cout << "  - YOLO系列 (v5/v8/v11)\n";
        std::cout << "  - SSD\n";
        std::cout << "  - RetinaNet\n";
        std::cout << "  - 其他目标检测模型\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    float conf_threshold = argc > 3 ? std::atof(argv[3]) : 0.25f;
    float iou_threshold = argc > 4 ? std::atof(argv[4]) : 0.45f;

    std::cout << "=== CRKIT 通用目标检测 ===" << std::endl;
    std::cout << "模型: " << model_path << std::endl;
    std::cout << "图像: " << image_path << std::endl;
    std::cout << "置信度阈值: " << conf_threshold << std::endl;
    std::cout << "NMS IOU阈值: " << iou_threshold << std::endl;
    std::cout << std::endl;

    // 1. 初始化SDK
    crkit::Initialize();
    crkit::Logger::Instance().SetLogLevel(crkit::LogLevel::INFO);

    // 2. 配置目标检测后处理器
    crkit::postprocess::DetectionConfig det_config;
    det_config.num_classes = 80;  // COCO数据集
    det_config.conf_threshold = conf_threshold;
    det_config.iou_threshold = iou_threshold;
    det_config.max_detections = 300;
    det_config.input_width = 640;
    det_config.input_height = 640;
    det_config.class_names = GetCOCOClassNames();

    // 设置输出格式（根据实际模型调整）
    // TRANSPOSED: [batch, channels, num_boxes] - YOLO v8/v11常用
    // FLAT: [batch, num_boxes, channels] - YOLO v5等
    det_config.output_format = crkit::postprocess::DetectionConfig::OutputFormat::TRANSPOSED;

    auto det_postprocessor =
        std::make_shared<crkit::postprocess::ObjectDetectionPostProcessor>(det_config);

    // 3. 配置推理引擎
    crkit::InferenceConfig config;
    config.engine_type = crkit::EngineType::ONNXRUNTIME;
    config.device_type = crkit::DeviceType::CPU;
    config.num_threads = 4;
    config.max_batch_size = 1;

    // 4. 创建推理引擎
    std::cout << "加载模型..." << std::endl;
    auto engine = crkit::CreateInferenceEngine(
        config.engine_type,
        model_path,
        config
    );

    if (!engine) {
        std::cerr << "创建推理引擎失败" << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "模型加载成功" << std::endl;
    std::cout << "引擎: " << engine->GetEngineVersion() << std::endl;

    // 显示模型信息
    auto input_names = engine->GetInputNames();
    std::cout << "\n模型信息:" << std::endl;
    std::cout << "  输入数量: " << input_names.size() << std::endl;
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto shape = engine->GetInputShape(input_names[i]);
        std::cout << "    - " << input_names[i] << ": [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 5. 预热
    std::cout << "\n预热中..." << std::endl;
    engine->WarmUp(5);
    std::cout << "预热完成" << std::endl;

    // 6. 加载并预处理图像
    std::cout << "\n加载图像: " << image_path << std::endl;

    crkit::utils::ImageLoadOptions load_options;
    load_options.target_width = 640;
    load_options.target_height = 640;
    load_options.normalize = false;
    load_options.hwc_to_chw = true;

    crkit::Tensor input_image = crkit::utils::LoadImage(image_path, load_options);

    if (input_image.Empty()) {
        std::cerr << "加载图像失败: " << image_path << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "图像加载成功" << std::endl;

    // 添加batch维度
    const auto& shape = input_image.GetShape();
    crkit::Tensor batch_input({1, shape[0], shape[1], shape[2]}, crkit::DataType::FLOAT32);
    std::memcpy(batch_input.Data(), input_image.Data(), input_image.ByteSize());

    // 7. 执行推理
    std::cout << "\n执行推理..." << std::endl;
    crkit::InferenceResult raw_result;
    auto status = engine->Infer(batch_input, raw_result);

    if (!status.IsOK()) {
        std::cerr << "推理失败: " << status.Message() << std::endl;
        crkit::Shutdown();
        return 1;
    }

    std::cout << "推理完成，耗时: " << raw_result.GetInferenceTime() << " ms" << std::endl;

    // 8. 后处理
    std::cout << "\n后处理中..." << std::endl;
    crkit::InferenceResult final_result;
    status = det_postprocessor->Process(raw_result.raw_outputs, final_result);

    if (!status.IsOK()) {
        std::cerr << "后处理失败: " << status.Message() << std::endl;
        crkit::Shutdown();
        return 1;
    }

    final_result.SetInferenceTime(raw_result.GetInferenceTime());

    // 9. 输出检测结果
    std::cout << "\n=== 检测结果 ===" << std::endl;
    std::cout << "检测数量: " << final_result.detections.size() << std::endl;
    std::cout << "推理时间: " << final_result.GetInferenceTime() << " ms" << std::endl;
    std::cout << "FPS: " << (1000.0f / final_result.GetInferenceTime()) << std::endl;

    if (final_result.detections.empty()) {
        std::cout << "\n未检测到任何目标!" << std::endl;
    } else {
        std::cout << "\n检测到的目标:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::setw(5) << "序号"
                  << std::setw(20) << "类别"
                  << std::setw(12) << "置信度"
                  << std::setw(15) << "X"
                  << std::setw(10) << "Y"
                  << std::setw(10) << "宽度"
                  << std::setw(10) << "高度" << std::endl;
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

        // 10. 可视化结果
        std::cout << "\n保存检测结果..." << std::endl;

        crkit::utils::ImageLoadOptions viz_options;
        viz_options.target_width = 640;
        viz_options.target_height = 640;
        viz_options.normalize = false;
        viz_options.hwc_to_chw = false;

        crkit::Tensor viz_image = crkit::utils::LoadImage(image_path, viz_options);
        crkit::utils::ImageUtils::DrawDetections(viz_image, final_result.detections, 2);

        std::string output_path = "detection_result.jpg";
        auto save_status = crkit::utils::ImageUtils::SaveImage(output_path, viz_image);

        if (save_status.IsOK()) {
            std::cout << "结果已保存至: " << output_path << std::endl;
        }

        // 统计各类别
        std::map<std::string, int> class_counts;
        for (const auto& det : final_result.detections) {
            class_counts[det.label]++;
        }

        std::cout << "\n各类别统计:" << std::endl;
        for (const auto& [label, count] : class_counts) {
            std::cout << "  " << label << ": " << count << std::endl;
        }
    }

    // 11. 清理资源
    engine->Release();
    crkit::Shutdown();

    std::cout << "\n完成!" << std::endl;
    return 0;
}
