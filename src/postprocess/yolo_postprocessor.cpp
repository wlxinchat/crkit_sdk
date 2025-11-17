#include "crkit/postprocess/yolo_postprocessor.h"
#include "crkit/utils/logger.h"
#include <algorithm>
#include <cmath>

namespace crkit {
namespace postprocess {

YOLOPostProcessor::YOLOPostProcessor(const YOLOConfig& config)
    : config_(config) {
}

Status YOLOPostProcessor::Process(const std::vector<Tensor>& model_outputs,
                                  InferenceResult& result) {
    if (model_outputs.empty()) {
        return Status(StatusCode::ERROR_INVALID_OUTPUT, "Empty model outputs");
    }

    result.Clear();
    std::vector<Detection> all_detections;

    // 解析YOLO输出
    // YOLO11/v8输出格式通常是: [batch, 84, 8400] 或 [batch, num_boxes, 4+num_classes]
    const Tensor& output = model_outputs[0];
    const auto& shape = output.GetShape();

    CRKIT_LOG_DEBUG("YOLO output shape: [", shape[0], ", ", shape[1], ", ", shape[2], "]");

    // 判断输出格式并解析
    if (shape.size() == 3) {
        // 格式1: [batch, num_classes+4, num_boxes] - YOLO v8/v11常见格式
        // 格式2: [batch, num_boxes, num_classes+4] - YOLO v5格式
        if (shape[1] > shape[2]) {
            // 格式2
            auto status = ParseYOLOOutput(output, all_detections);
            if (!status.IsOK()) return status;
        } else {
            // 格式1 - 需要转置
            auto status = ParseYOLO11Output(output, all_detections);
            if (!status.IsOK()) return status;
        }
    }

    // 应用NMS
    ApplyNMS(all_detections);

    // 过滤低置信度检测
    for (auto& det : all_detections) {
        if (det.confidence >= config_.conf_threshold) {
            result.AddDetection(det);
        }
    }

    // 限制最大检测数量
    if (result.detections.size() > static_cast<size_t>(config_.max_detections)) {
        result.SortDetectionsByConfidence();
        result.detections.resize(config_.max_detections);
    }

    CRKIT_LOG_INFO("YOLO post-processing: ", result.detections.size(), " detections");

    return Status();
}

Status YOLOPostProcessor::ParseYOLOOutput(const Tensor& output,
                                         std::vector<Detection>& detections) {
    const auto& shape = output.GetShape();
    // 期望形状: [batch, num_boxes, 4+num_classes]
    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_OUTPUT, "Invalid YOLO output shape");
    }

    int num_boxes = shape[1];
    int box_dim = shape[2];  // 4 (bbox) + num_classes

    if (box_dim < 5) {
        return Status(StatusCode::ERROR_INVALID_OUTPUT, "Invalid box dimension");
    }

    const float* data = output.Data<float>();

    for (int i = 0; i < num_boxes; ++i) {
        const float* box_data = data + i * box_dim;

        // 解析边界框 (cx, cy, w, h)
        float cx = box_data[0];
        float cy = box_data[1];
        float w = box_data[2];
        float h = box_data[3];

        // 转换为 (x, y, w, h) 格式
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        // 查找最高置信度的类别
        int best_class_id = -1;
        float best_confidence = 0.0f;

        for (int c = 0; c < config_.num_classes; ++c) {
            float class_score = box_data[4 + c];
            if (class_score > best_confidence) {
                best_confidence = class_score;
                best_class_id = c;
            }
        }

        // 过滤低置信度
        if (best_confidence < config_.conf_threshold) {
            continue;
        }

        // 创建检测结果
        Detection det;
        det.bbox = BBox(x, y, w, h);
        det.class_id = best_class_id;
        det.confidence = best_confidence;

        // 设置类别名称
        if (best_class_id < static_cast<int>(config_.class_names.size())) {
            det.label = config_.class_names[best_class_id];
        } else {
            det.label = "class_" + std::to_string(best_class_id);
        }

        detections.push_back(det);
    }

    return Status();
}

Status YOLOPostProcessor::ParseYOLO11Output(const Tensor& output,
                                           std::vector<Detection>& detections) {
    const auto& shape = output.GetShape();
    // 期望形状: [batch, 4+num_classes, num_boxes]
    // 例如: [1, 84, 8400] for COCO (80 classes)

    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_OUTPUT, "Invalid YOLO11 output shape");
    }

    int batch = shape[0];
    int box_dim = shape[1];  // 4 + num_classes
    int num_boxes = shape[2];

    if (box_dim < 5 || batch != 1) {
        return Status(StatusCode::ERROR_INVALID_OUTPUT,
                     "Invalid YOLO11 output dimensions");
    }

    const float* data = output.Data<float>();

    // YOLO11输出格式: [cx, cy, w, h, class1, class2, ..., classN]
    // 数据布局: [batch, channels, boxes] - 转置格式

    for (int i = 0; i < num_boxes; ++i) {
        // 提取边界框坐标 (前4个通道)
        float cx = data[0 * num_boxes + i];
        float cy = data[1 * num_boxes + i];
        float w = data[2 * num_boxes + i];
        float h = data[3 * num_boxes + i];

        // 转换为 (x, y, w, h) 格式
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        // 查找最高置信度的类别
        int best_class_id = -1;
        float best_confidence = 0.0f;

        for (int c = 0; c < config_.num_classes; ++c) {
            float class_score = data[(4 + c) * num_boxes + i];
            if (class_score > best_confidence) {
                best_confidence = class_score;
                best_class_id = c;
            }
        }

        // 过滤低置信度
        if (best_confidence < config_.conf_threshold) {
            continue;
        }

        // 创建检测结果
        Detection det;
        det.bbox = BBox(x, y, w, h);
        det.class_id = best_class_id;
        det.confidence = best_confidence;

        // 设置类别名称
        if (best_class_id < static_cast<int>(config_.class_names.size())) {
            det.label = config_.class_names[best_class_id];
        } else {
            det.label = "class_" + std::to_string(best_class_id);
        }

        detections.push_back(det);
    }

    return Status();
}

void YOLOPostProcessor::ApplyNMS(std::vector<Detection>& detections) {
    NMS::Apply(detections, config_.iou_threshold, config_.max_detections);
}

float YOLOPostProcessor::ComputeIOU(const BBox& a, const BBox& b) {
    return NMS::ComputeIOU(a, b);
}

void YOLOPostProcessor::ScaleCoordinates(Detection& det,
                                        int orig_width,
                                        int orig_height) {
    float scale_x = static_cast<float>(orig_width) / config_.input_width;
    float scale_y = static_cast<float>(orig_height) / config_.input_height;

    det.bbox.x *= scale_x;
    det.bbox.y *= scale_y;
    det.bbox.width *= scale_x;
    det.bbox.height *= scale_y;
}

// NMS实现
void NMS::Apply(std::vector<Detection>& detections,
               float iou_threshold,
               int max_detections) {
    if (detections.empty()) return;

    // 按置信度排序（降序）
    std::sort(detections.begin(), detections.end(),
             [](const Detection& a, const Detection& b) {
                 return a.confidence > b.confidence;
             });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        if (max_detections > 0 && static_cast<int>(result.size()) >= max_detections) {
            break;
        }

        // 抑制与当前检测重叠度高的其他检测
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            // 只对同一类别应用NMS
            if (detections[i].class_id != detections[j].class_id) {
                continue;
            }

            float iou = ComputeIOU(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(result);
}

float NMS::ComputeIOU(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float intersection_width = std::max(0.0f, x2 - x1);
    float intersection_height = std::max(0.0f, y2 - y1);
    float intersection_area = intersection_width * intersection_height;

    float area_a = a.Area();
    float area_b = b.Area();
    float union_area = area_a + area_b - intersection_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection_area / union_area;
}

} // namespace postprocess
} // namespace crkit
