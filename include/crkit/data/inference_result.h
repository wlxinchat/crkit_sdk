#pragma once

#include "crkit/data/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace crkit {

// 边界框
struct BBox {
    float x;       // 左上角x坐标
    float y;       // 左上角y坐标
    float width;   // 宽度
    float height;  // 高度

    BBox() : x(0), y(0), width(0), height(0) {}
    BBox(float x_, float y_, float w, float h)
        : x(x_), y(y_), width(w), height(h) {}

    // 计算面积
    float Area() const { return width * height; }

    // 中心点坐标
    float CenterX() const { return x + width / 2; }
    float CenterY() const { return y + height / 2; }
};

// 关键点
struct KeyPoint {
    float x;
    float y;
    float confidence;

    KeyPoint() : x(0), y(0), confidence(0) {}
    KeyPoint(float x_, float y_, float conf = 1.0f)
        : x(x_), y(y_), confidence(conf) {}
};

// 检测结果（目标检测、缺陷检测）
struct Detection {
    BBox bbox;                      // 边界框
    int class_id;                   // 类别ID
    std::string label;              // 类别标签
    float confidence;               // 置信度
    std::vector<KeyPoint> keypoints; // 关键点（可选）
    std::map<std::string, float> attributes; // 其他属性

    Detection() : class_id(-1), confidence(0.0f) {}
};

// 分类结果
struct Classification {
    int class_id;                   // 类别ID
    std::string label;              // 类别标签
    float confidence;               // 置信度

    Classification() : class_id(-1), confidence(0.0f) {}
    Classification(int id, const std::string& lbl, float conf)
        : class_id(id), label(lbl), confidence(conf) {}
};

// 分割结果
struct Segmentation {
    Tensor mask;                    // 分割掩码（HxW或NxHxW）
    std::vector<int> class_ids;     // 类别ID列表
    std::vector<std::string> labels; // 类别标签列表
};

// 推理结果（通用结构）
class InferenceResult {
public:
    InferenceResult() : inference_time_ms_(0.0f) {}

    // 检测结果
    std::vector<Detection> detections;

    // 分类结果
    std::vector<Classification> classifications;

    // 分割结果
    Segmentation segmentation;

    // 原始输出张量（用于自定义后处理）
    std::vector<Tensor> raw_outputs;

    // 推理时间（毫秒）
    float inference_time_ms_;

    // 额外的元数据
    std::map<std::string, std::string> metadata;

    // 设置推理时间
    void SetInferenceTime(float time_ms) { inference_time_ms_ = time_ms; }

    // 获取推理时间
    float GetInferenceTime() const { return inference_time_ms_; }

    // 添加检测结果
    void AddDetection(const Detection& det) {
        detections.push_back(det);
    }

    // 添加分类结果
    void AddClassification(const Classification& cls) {
        classifications.push_back(cls);
    }

    // 按置信度排序检测结果
    void SortDetectionsByConfidence() {
        std::sort(detections.begin(), detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });
    }

    // 过滤低置信度检测结果
    void FilterDetections(float min_confidence) {
        detections.erase(
            std::remove_if(detections.begin(), detections.end(),
                          [min_confidence](const Detection& det) {
                              return det.confidence < min_confidence;
                          }),
            detections.end()
        );
    }

    // 获取最高置信度的分类结果
    Classification GetTopClassification() const {
        if (classifications.empty()) {
            return Classification();
        }
        return *std::max_element(classifications.begin(), classifications.end(),
                                [](const Classification& a, const Classification& b) {
                                    return a.confidence < b.confidence;
                                });
    }

    // 清空结果
    void Clear() {
        detections.clear();
        classifications.clear();
        segmentation.mask.Clear();
        segmentation.class_ids.clear();
        segmentation.labels.clear();
        raw_outputs.clear();
        metadata.clear();
        inference_time_ms_ = 0.0f;
    }

    // 判断是否为空
    bool Empty() const {
        return detections.empty() &&
               classifications.empty() &&
               segmentation.mask.Empty();
    }
};

} // namespace crkit
