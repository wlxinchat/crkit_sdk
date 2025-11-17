#pragma once

#include "crkit/data/tensor.h"
#include "crkit/core/types.h"
#include <string>

namespace crkit {
namespace utils {

// 图像加载选项
struct ImageLoadOptions {
    int target_width = -1;   // -1表示保持原始尺寸
    int target_height = -1;
    bool normalize = false;
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> std = {1.0f, 1.0f, 1.0f};
    bool hwc_to_chw = false; // HWC转CHW (OpenCV默认HWC, 大部分模型需要CHW)
};

// 图像工具类
class ImageUtils {
public:
    // 从文件加载图像
    static Status LoadImage(const std::string& filepath,
                           Tensor& output,
                           const ImageLoadOptions& options = ImageLoadOptions());

    // 从内存加载图像
    static Status LoadImageFromMemory(const uint8_t* data,
                                     size_t size,
                                     Tensor& output,
                                     const ImageLoadOptions& options = ImageLoadOptions());

    // 保存图像
    static Status SaveImage(const std::string& filepath, const Tensor& image);

    // 调整图像大小
    static Status ResizeImage(const Tensor& input,
                             Tensor& output,
                             int width,
                             int height,
                             bool keep_aspect_ratio = true);

    // 裁剪图像
    static Status CropImage(const Tensor& input,
                           Tensor& output,
                           int x, int y,
                           int width, int height);

    // 绘制边界框
    static Status DrawBoundingBox(Tensor& image,
                                 const BBox& bbox,
                                 const std::string& label = "",
                                 float confidence = 0.0f,
                                 int thickness = 2);

    // 绘制检测结果
    static Status DrawDetections(Tensor& image,
                                const std::vector<Detection>& detections,
                                int thickness = 2);

    // 批量加载图像
    static Status LoadImageBatch(const std::vector<std::string>& filepaths,
                                std::vector<Tensor>& outputs,
                                const ImageLoadOptions& options = ImageLoadOptions());

    // 图像格式转换
    static Status ConvertColorSpace(const Tensor& input,
                                   Tensor& output,
                                   const std::string& from,
                                   const std::string& to);

    // HWC到CHW转换
    static Status HWCToCHW(const Tensor& input, Tensor& output);

    // CHW到HWC转换
    static Status CHWToHWC(const Tensor& input, Tensor& output);
};

// 便捷函数
inline Tensor LoadImage(const std::string& filepath,
                       const ImageLoadOptions& options = ImageLoadOptions()) {
    Tensor output;
    ImageUtils::LoadImage(filepath, output, options);
    return output;
}

} // namespace utils
} // namespace crkit
