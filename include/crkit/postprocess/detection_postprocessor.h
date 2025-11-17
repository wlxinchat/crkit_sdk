#pragma once

#include "crkit/core/inference_context.h"
#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"
#include <vector>
#include <string>

namespace crkit {
namespace postprocess {

// 目标检测后处理配置
struct DetectionConfig {
    int num_classes = 80;                    // 类别数量
    float conf_threshold = 0.25f;            // 置信度阈值
    float iou_threshold = 0.45f;             // NMS的IOU阈值
    int max_detections = 300;                // 最大检测数量
    std::vector<std::string> class_names;    // 类别名称
    int input_width = 640;                   // 输入图像宽度
    int input_height = 640;                  // 输入图像高度

    // 输出格式配置
    enum class OutputFormat {
        FLAT,       // [batch, num_boxes, 4+1+num_classes] - bbox + objectness + classes
        TRANSPOSED, // [batch, 4+num_classes, num_boxes] - 转置格式
        SEPARATE    // 独立的bbox和class输出
    };
    OutputFormat output_format = OutputFormat::TRANSPOSED;
};

// 通用目标检测后处理器
class ObjectDetectionPostProcessor : public IPostProcessor {
public:
    explicit ObjectDetectionPostProcessor(const DetectionConfig& config);
    ~ObjectDetectionPostProcessor() override = default;

    // 后处理主函数
    Status Process(const std::vector<Tensor>& model_outputs,
                  InferenceResult& result) override;

    // 设置配置
    void SetConfig(const DetectionConfig& config) { config_ = config; }
    const DetectionConfig& GetConfig() const { return config_; }

private:
    DetectionConfig config_;

    // 解析扁平格式输出 [batch, num_boxes, 4+1+num_classes]
    Status ParseFlatOutput(const Tensor& output,
                          std::vector<Detection>& detections);

    // 解析转置格式输出 [batch, 4+num_classes, num_boxes]
    Status ParseTransposedOutput(const Tensor& output,
                                std::vector<Detection>& detections);

    // NMS（非极大值抑制）
    void ApplyNMS(std::vector<Detection>& detections);

    // 计算IOU
    float ComputeIOU(const BBox& a, const BBox& b);

    // 缩放坐标到原始图像尺寸
    void ScaleCoordinates(Detection& det, int orig_width, int orig_height);
};

// NMS实现（可独立使用）
class NMS {
public:
    static void Apply(std::vector<Detection>& detections,
                     float iou_threshold,
                     int max_detections = -1);

    static float ComputeIOU(const BBox& a, const BBox& b);
};

} // namespace postprocess
} // namespace crkit
