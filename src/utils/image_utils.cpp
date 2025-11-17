#include "crkit/utils/image_utils.h"
#include "crkit/utils/logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace crkit {
namespace utils {

Status ImageUtils::LoadImage(const std::string& filepath,
                             Tensor& output,
                             const ImageLoadOptions& options) {
    // 加载图像
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT,
                     "Failed to load image: " + filepath);
    }

    // 转换为RGB（OpenCV默认BGR）
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize
    if (options.target_width > 0 && options.target_height > 0) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(options.target_width, options.target_height));
        img = resized;
    }

    // 转换为float
    img.convertTo(img, CV_32FC3);

    // 归一化
    if (options.normalize) {
        for (int c = 0; c < 3; ++c) {
            float mean_val = c < options.mean.size() ? options.mean[c] : 0.0f;
            float std_val = c < options.std.size() ? options.std[c] : 1.0f;

            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    img.at<cv::Vec3f>(i, j)[c] =
                        (img.at<cv::Vec3f>(i, j)[c] / 255.0f - mean_val) / std_val;
                }
            }
        }
    } else {
        img /= 255.0f;  // 归一化到[0,1]
    }

    // HWC to CHW转换
    if (options.hwc_to_chw) {
        Status status = HWCToCHW(
            Tensor(img.data, {img.rows, img.cols, 3}, DataType::FLOAT32, true),
            output
        );
        return status;
    } else {
        // 创建输出tensor (HWC格式)
        Shape shape = {img.rows, img.cols, 3};
        output = Tensor(shape, DataType::FLOAT32);
        std::memcpy(output.Data(), img.data, img.total() * img.elemSize());
    }

    return Status();
}

Status ImageUtils::LoadImageFromMemory(const uint8_t* data,
                                       size_t size,
                                       Tensor& output,
                                       const ImageLoadOptions& options) {
    std::vector<uint8_t> buffer(data, data + size);
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (img.empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT,
                     "Failed to decode image from memory");
    }

    // 后续处理与LoadImage相同
    // TODO: 复用代码
    return Status();
}

Status ImageUtils::SaveImage(const std::string& filepath, const Tensor& image) {
    if (image.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty tensor");
    }

    const auto& shape = image.GetShape();
    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Invalid image shape");
    }

    int height = shape[0];
    int width = shape[1];
    int channels = shape[2];

    // 创建OpenCV Mat
    cv::Mat img(height, width, CV_32FC3, const_cast<void*>(image.Data()));

    // 反归一化
    img *= 255.0f;

    // 转换为uint8
    img.convertTo(img, CV_8UC3);

    // BGR转换
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    // 保存
    bool success = cv::imwrite(filepath, img);
    if (!success) {
        return Status(StatusCode::ERROR_UNKNOWN, "Failed to save image");
    }

    return Status();
}

Status ImageUtils::ResizeImage(const Tensor& input,
                               Tensor& output,
                               int width,
                               int height,
                               bool keep_aspect_ratio) {
    if (input.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty input tensor");
    }

    const auto& shape = input.GetShape();
    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Invalid input shape");
    }

    int src_height = shape[0];
    int src_width = shape[1];
    int channels = shape[2];

    cv::Mat src(src_height, src_width, CV_32FC3, const_cast<void*>(input.Data()));
    cv::Mat dst;

    if (keep_aspect_ratio) {
        // Letterbox resize - 保持宽高比，填充到目标尺寸
        float scale = std::min(
            static_cast<float>(width) / src_width,
            static_cast<float>(height) / src_height
        );

        int new_width = static_cast<int>(src_width * scale);
        int new_height = static_cast<int>(src_height * scale);

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_width, new_height));

        // 创建目标图像并填充
        dst = cv::Mat(height, width, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
        int top = (height - new_height) / 2;
        int left = (width - new_width) / 2;
        resized.copyTo(dst(cv::Rect(left, top, new_width, new_height)));
    } else {
        cv::resize(src, dst, cv::Size(width, height));
    }

    // 创建输出tensor
    output = Tensor({height, width, channels}, DataType::FLOAT32);
    std::memcpy(output.Data(), dst.data, dst.total() * dst.elemSize());

    return Status();
}

Status ImageUtils::CropImage(const Tensor& input,
                             Tensor& output,
                             int x, int y,
                             int width, int height) {
    if (input.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty input tensor");
    }

    const auto& shape = input.GetShape();
    int src_height = shape[0];
    int src_width = shape[1];
    int channels = shape[2];

    if (x + width > src_width || y + height > src_height) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Crop region out of bounds");
    }

    cv::Mat src(src_height, src_width, CV_32FC3, const_cast<void*>(input.Data()));
    cv::Mat dst = src(cv::Rect(x, y, width, height)).clone();

    output = Tensor({height, width, channels}, DataType::FLOAT32);
    std::memcpy(output.Data(), dst.data, dst.total() * dst.elemSize());

    return Status();
}

Status ImageUtils::DrawBoundingBox(Tensor& image,
                                   const BBox& bbox,
                                   const std::string& label,
                                   float confidence,
                                   int thickness) {
    if (image.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty image");
    }

    const auto& shape = image.GetShape();
    cv::Mat img(shape[0], shape[1], CV_32FC3, image.Data());

    // 转换为uint8以便绘制
    cv::Mat img_u8;
    img.convertTo(img_u8, CV_8UC3, 255.0);

    // 绘制矩形
    cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
    cv::rectangle(img_u8, rect, cv::Scalar(0, 255, 0), thickness);

    // 绘制标签
    if (!label.empty()) {
        std::string text = label;
        if (confidence > 0) {
            char conf_str[16];
            snprintf(conf_str, sizeof(conf_str), " %.2f", confidence);
            text += conf_str;
        }

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                             0.5, 1, &baseline);

        cv::rectangle(img_u8,
                     cv::Point(bbox.x, bbox.y - text_size.height - 5),
                     cv::Point(bbox.x + text_size.width, bbox.y),
                     cv::Scalar(0, 255, 0), -1);

        cv::putText(img_u8, text,
                   cv::Point(bbox.x, bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(0, 0, 0), 1);
    }

    // 转换回float
    img_u8.convertTo(img, CV_32FC3, 1.0 / 255.0);

    return Status();
}

Status ImageUtils::DrawDetections(Tensor& image,
                                  const std::vector<Detection>& detections,
                                  int thickness) {
    for (const auto& det : detections) {
        auto status = DrawBoundingBox(image, det.bbox, det.label,
                                      det.confidence, thickness);
        if (!status.IsOK()) {
            return status;
        }
    }
    return Status();
}

Status ImageUtils::LoadImageBatch(const std::vector<std::string>& filepaths,
                                  std::vector<Tensor>& outputs,
                                  const ImageLoadOptions& options) {
    outputs.clear();
    outputs.reserve(filepaths.size());

    for (const auto& filepath : filepaths) {
        Tensor tensor;
        auto status = LoadImage(filepath, tensor, options);
        if (!status.IsOK()) {
            CRKIT_LOG_WARNING("Failed to load image: ", filepath);
            continue;
        }
        outputs.push_back(std::move(tensor));
    }

    return Status();
}

Status ImageUtils::ConvertColorSpace(const Tensor& input,
                                     Tensor& output,
                                     const std::string& from,
                                     const std::string& to) {
    // 简化实现
    output = input.Clone();
    return Status();
}

Status ImageUtils::HWCToCHW(const Tensor& input, Tensor& output) {
    if (input.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty input");
    }

    const auto& shape = input.GetShape();
    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Input must be 3D (HWC)");
    }

    int H = shape[0];
    int W = shape[1];
    int C = shape[2];

    // 创建输出tensor (CHW格式)
    output = Tensor({C, H, W}, input.GetDataType());

    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    // HWC -> CHW转换
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                dst[c * H * W + h * W + w] = src[h * W * C + w * C + c];
            }
        }
    }

    return Status();
}

Status ImageUtils::CHWToHWC(const Tensor& input, Tensor& output) {
    if (input.Empty()) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Empty input");
    }

    const auto& shape = input.GetShape();
    if (shape.size() != 3) {
        return Status(StatusCode::ERROR_INVALID_INPUT, "Input must be 3D (CHW)");
    }

    int C = shape[0];
    int H = shape[1];
    int W = shape[2];

    // 创建输出tensor (HWC格式)
    output = Tensor({H, W, C}, input.GetDataType());

    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    // CHW -> HWC转换
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                dst[h * W * C + w * C + c] = src[c * H * W + h * W + w];
            }
        }
    }

    return Status();
}

} // namespace utils
} // namespace crkit
