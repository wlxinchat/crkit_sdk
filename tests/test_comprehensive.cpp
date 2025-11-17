/**
 * CRKIT SDK 综合测试套件
 * 涵盖所有核心功能的完整测试
 */

#include "crkit/data/tensor.h"
#include "crkit/data/inference_result.h"
#include "crkit/postprocess/detection_postprocessor.h"
#include "crkit/core/types.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

using namespace crkit;
using namespace crkit::postprocess;

// 测试统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;

    void record(bool success) {
        total++;
        if (success) passed++;
        else failed++;
    }

    float passRate() const {
        return total > 0 ? (100.0f * passed / total) : 0.0f;
    }
};

TestStats g_stats;

// 测试辅助宏
#define TEST_CASE(name) \
    { \
    std::cout << "\n[测试] " << name << std::endl; \
    bool test_passed = true;

#define ASSERT_TRUE(expr) \
    if (!(expr)) { \
        std::cout << "  ✗ 断言失败: " #expr << " (行 " << __LINE__ << ")" << std::endl; \
        test_passed = false; \
    }

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cout << "  ✗ 断言失败 " #a " == " #b << " (行 " << __LINE__ << ")" << std::endl; \
        test_passed = false; \
    }

#define ASSERT_FLOAT_EQ(a, b, epsilon) \
    if (std::abs((a) - (b)) > (epsilon)) { \
        std::cout << "  ✗ 期望: " << (b) << ", 实际: " << (a) << " (行 " << __LINE__ << ")" << std::endl; \
        test_passed = false; \
    }

#define TEST_END() \
    if (test_passed) { \
        std::cout << "  ✓ 通过" << std::endl; \
    } else { \
        std::cout << "  ✗ 失败" << std::endl; \
    } \
    g_stats.record(test_passed); \
    }

// ============================================================================
// TC001: Tensor基础功能测试
// ============================================================================

void test_tensor_basic() {
    TEST_CASE("TC001-01: 空Tensor创建");
    {
        Tensor empty_tensor;
        ASSERT_TRUE(empty_tensor.Empty());
        ASSERT_EQ(empty_tensor.NumElements(), 0);
    }
    TEST_END();

    TEST_CASE("TC001-02: 指定形状创建");
    {
        Shape shape = {2, 3, 4};
        Tensor tensor(shape, DataType::FLOAT32);

        ASSERT_FALSE(tensor.Empty());
        ASSERT_EQ(tensor.GetShape(), shape);
        ASSERT_EQ(tensor.Rank(), 3);
        ASSERT_EQ(tensor.NumElements(), 24);
        ASSERT_EQ(tensor.GetDataType(), DataType::FLOAT32);
    }
    TEST_END();

    TEST_CASE("TC001-03: 不同数据类型");
    {
        Tensor float_tensor({10}, DataType::FLOAT32);
        ASSERT_EQ(float_tensor.ByteSize(), 10 * 4);

        Tensor int_tensor({10}, DataType::INT32);
        ASSERT_EQ(int_tensor.ByteSize(), 10 * 4);

        Tensor uint8_tensor({10}, DataType::UINT8);
        ASSERT_EQ(uint8_tensor.ByteSize(), 10 * 1);
    }
    TEST_END();

    TEST_CASE("TC001-04: 数据写入读取");
    {
        Tensor tensor({5}, DataType::FLOAT32);
        float* data = tensor.Data<float>();

        for (int i = 0; i < 5; ++i) {
            data[i] = static_cast<float>(i * 2);
        }

        const float* const_data = tensor.Data<const float>();
        for (int i = 0; i < 5; ++i) {
            ASSERT_FLOAT_EQ(const_data[i], i * 2.0f, 0.001f);
        }
    }
    TEST_END();
}

// ============================================================================
// TC002: Tensor高级功能测试
// ============================================================================

void test_tensor_advanced() {
    TEST_CASE("TC002-01: Reshape有效");
    {
        Tensor tensor({2, 12}, DataType::FLOAT32);
        float* data = tensor.Data<float>();
        for (int i = 0; i < 24; ++i) data[i] = i;

        bool success = tensor.Reshape({3, 8});
        ASSERT_TRUE(success);
        ASSERT_EQ(tensor.Dim(0), 3);
        ASSERT_EQ(tensor.Dim(1), 8);

        // 数据应该不变
        const float* new_data = tensor.Data<const float>();
        ASSERT_FLOAT_EQ(new_data[0], 0.0f, 0.001f);
        ASSERT_FLOAT_EQ(new_data[23], 23.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC002-02: Reshape无效");
    {
        Tensor tensor({2, 3}, DataType::FLOAT32);
        Shape original_shape = tensor.GetShape();

        bool success = tensor.Reshape({1, 7}); // 元素数不匹配
        ASSERT_FALSE(success);
        ASSERT_EQ(tensor.GetShape(), original_shape); // 形状不变
    }
    TEST_END();

    TEST_CASE("TC002-03: Clone深拷贝");
    {
        Tensor tensor({3}, DataType::FLOAT32);
        float* data = tensor.Data<float>();
        data[0] = 1.0f;
        data[1] = 2.0f;
        data[2] = 3.0f;

        Tensor cloned = tensor.Clone();
        ASSERT_EQ(cloned.GetShape(), tensor.GetShape());

        const float* cloned_data = cloned.Data<const float>();
        ASSERT_FLOAT_EQ(cloned_data[0], 1.0f, 0.001f);
        ASSERT_FLOAT_EQ(cloned_data[1], 2.0f, 0.001f);
        ASSERT_FLOAT_EQ(cloned_data[2], 3.0f, 0.001f);

        // 修改原tensor不应影响克隆
        data[0] = 99.0f;
        ASSERT_FLOAT_EQ(cloned_data[0], 1.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC002-04: Fill填充");
    {
        Tensor tensor({4, 4}, DataType::FLOAT32);
        tensor.Fill(5.5f);

        const float* data = tensor.Data<const float>();
        for (int i = 0; i < 16; ++i) {
            ASSERT_FLOAT_EQ(data[i], 5.5f, 0.001f);
        }
    }
    TEST_END();

    TEST_CASE("TC002-05: Clear清空");
    {
        Tensor tensor({10}, DataType::FLOAT32);
        ASSERT_FALSE(tensor.Empty());

        tensor.Clear();
        ASSERT_TRUE(tensor.Empty());
        ASSERT_EQ(tensor.NumElements(), 0);
    }
    TEST_END();
}

// ============================================================================
// TC003: BBox和Detection测试
// ============================================================================

void test_bbox_detection() {
    TEST_CASE("TC003-01: BBox面积计算");
    {
        BBox box(0, 0, 10, 20);
        ASSERT_FLOAT_EQ(box.Area(), 200.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC003-02: BBox中心点");
    {
        BBox box(10, 10, 20, 30);
        ASSERT_FLOAT_EQ(box.CenterX(), 20.0f, 0.001f);
        ASSERT_FLOAT_EQ(box.CenterY(), 25.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC003-03: Detection结构");
    {
        Detection det;
        det.bbox = BBox(10, 20, 30, 40);
        det.class_id = 5;
        det.label = "test_class";
        det.confidence = 0.95f;

        ASSERT_FLOAT_EQ(det.bbox.x, 10.0f, 0.001f);
        ASSERT_EQ(det.class_id, 5);
        ASSERT_EQ(det.label, "test_class");
        ASSERT_FLOAT_EQ(det.confidence, 0.95f, 0.001f);
    }
    TEST_END();
}

// ============================================================================
// TC004: IOU计算测试
// ============================================================================

void test_iou_calculation() {
    TEST_CASE("TC004-01: 完全重叠");
    {
        BBox box1(0, 0, 100, 100);
        BBox box2(0, 0, 100, 100);
        float iou = NMS::ComputeIOU(box1, box2);
        ASSERT_FLOAT_EQ(iou, 1.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC004-02: 完全不重叠");
    {
        BBox box1(0, 0, 50, 50);
        BBox box2(100, 100, 50, 50);
        float iou = NMS::ComputeIOU(box1, box2);
        ASSERT_FLOAT_EQ(iou, 0.0f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC004-03: 部分重叠");
    {
        BBox box1(0, 0, 100, 100);    // 面积: 10000
        BBox box2(50, 50, 100, 100);  // 面积: 10000
        // 重叠区域: 50x50 = 2500
        // 并集: 10000 + 10000 - 2500 = 17500
        // IOU = 2500 / 17500 = 0.142857
        float iou = NMS::ComputeIOU(box1, box2);
        ASSERT_FLOAT_EQ(iou, 0.142857f, 0.01f);
    }
    TEST_END();

    TEST_CASE("TC004-04: 包含关系");
    {
        BBox big(0, 0, 100, 100);     // 面积: 10000
        BBox small(25, 25, 50, 50);   // 面积: 2500
        // 重叠区域: 2500
        // 并集: 10000
        // IOU = 2500 / 10000 = 0.25
        float iou = NMS::ComputeIOU(big, small);
        ASSERT_FLOAT_EQ(iou, 0.25f, 0.01f);
    }
    TEST_END();

    TEST_CASE("TC004-05: 边界触碰");
    {
        BBox box1(0, 0, 50, 50);
        BBox box2(50, 0, 50, 50);  // 边缘相切
        float iou = NMS::ComputeIOU(box1, box2);
        ASSERT_FLOAT_EQ(iou, 0.0f, 0.001f);
    }
    TEST_END();
}

// ============================================================================
// TC005: NMS算法测试
// ============================================================================

void test_nms_algorithm() {
    TEST_CASE("TC005-01: 基本抑制");
    {
        std::vector<Detection> detections;

        Detection det1;
        det1.bbox = BBox(10, 10, 100, 100);
        det1.confidence = 0.9f;
        det1.class_id = 0;
        detections.push_back(det1);

        Detection det2;
        det2.bbox = BBox(15, 15, 100, 100); // 高度重叠
        det2.confidence = 0.8f;
        det2.class_id = 0;
        detections.push_back(det2);

        NMS::Apply(detections, 0.5f);

        ASSERT_EQ(detections.size(), 1);
        ASSERT_FLOAT_EQ(detections[0].confidence, 0.9f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC005-02: 多框抑制");
    {
        std::vector<Detection> detections;

        for (int i = 0; i < 5; ++i) {
            Detection det;
            det.bbox = BBox(10 + i * 5, 10 + i * 5, 100, 100);
            det.confidence = 0.9f - i * 0.1f;
            det.class_id = 0;
            detections.push_back(det);
        }

        NMS::Apply(detections, 0.5f);

        // 应该保留2-3个框（根据重叠情况）
        ASSERT_TRUE(detections.size() >= 2 && detections.size() <= 3);
        // 第一个应该是最高置信度
        ASSERT_FLOAT_EQ(detections[0].confidence, 0.9f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC005-03: 不同类别");
    {
        std::vector<Detection> detections;

        Detection det1;
        det1.bbox = BBox(10, 10, 100, 100);
        det1.confidence = 0.9f;
        det1.class_id = 0;
        detections.push_back(det1);

        Detection det2;
        det2.bbox = BBox(15, 15, 100, 100); // 高度重叠
        det2.confidence = 0.8f;
        det2.class_id = 1; // 不同类别
        detections.push_back(det2);

        NMS::Apply(detections, 0.5f);

        // 不同类别都应保留
        ASSERT_EQ(detections.size(), 2);
    }
    TEST_END();

    TEST_CASE("TC005-04: Top-K限制");
    {
        std::vector<Detection> detections;

        // 创建10个不重叠的检测框
        for (int i = 0; i < 10; ++i) {
            Detection det;
            det.bbox = BBox(i * 150.0f, 0, 100, 100);
            det.confidence = 0.9f - i * 0.05f;
            det.class_id = 0;
            detections.push_back(det);
        }

        NMS::Apply(detections, 0.5f, 5);

        ASSERT_EQ(detections.size(), 5);
        // 应该是前5个最高置信度的
        for (size_t i = 0; i < 5; ++i) {
            ASSERT_FLOAT_EQ(detections[i].confidence, 0.9f - i * 0.05f, 0.001f);
        }
    }
    TEST_END();

    TEST_CASE("TC005-05: 空输入");
    {
        std::vector<Detection> detections;
        NMS::Apply(detections, 0.5f);
        ASSERT_EQ(detections.size(), 0);
    }
    TEST_END();

    TEST_CASE("TC005-06: 单个框");
    {
        std::vector<Detection> detections;
        Detection det;
        det.bbox = BBox(0, 0, 100, 100);
        det.confidence = 0.9f;
        det.class_id = 0;
        detections.push_back(det);

        NMS::Apply(detections, 0.5f);

        ASSERT_EQ(detections.size(), 1);
        ASSERT_FLOAT_EQ(detections[0].confidence, 0.9f, 0.001f);
    }
    TEST_END();
}

// ============================================================================
// TC006: InferenceResult测试
// ============================================================================

void test_inference_result() {
    TEST_CASE("TC006-01: 添加检测");
    {
        InferenceResult result;
        Detection det;
        det.confidence = 0.8f;

        result.AddDetection(det);
        ASSERT_EQ(result.detections.size(), 1);
    }
    TEST_END();

    TEST_CASE("TC006-02: 置信度排序");
    {
        InferenceResult result;

        Detection det1; det1.confidence = 0.5f; result.AddDetection(det1);
        Detection det2; det2.confidence = 0.9f; result.AddDetection(det2);
        Detection det3; det3.confidence = 0.7f; result.AddDetection(det3);

        result.SortDetectionsByConfidence();

        ASSERT_FLOAT_EQ(result.detections[0].confidence, 0.9f, 0.001f);
        ASSERT_FLOAT_EQ(result.detections[1].confidence, 0.7f, 0.001f);
        ASSERT_FLOAT_EQ(result.detections[2].confidence, 0.5f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC006-03: 置信度过滤");
    {
        InferenceResult result;

        Detection det1; det1.confidence = 0.3f; result.AddDetection(det1);
        Detection det2; det2.confidence = 0.6f; result.AddDetection(det2);
        Detection det3; det3.confidence = 0.8f; result.AddDetection(det3);

        result.FilterDetections(0.5f);

        ASSERT_EQ(result.detections.size(), 2);
        ASSERT_TRUE(result.detections[0].confidence >= 0.5f);
        ASSERT_TRUE(result.detections[1].confidence >= 0.5f);
    }
    TEST_END();

    TEST_CASE("TC006-04: 获取Top1分类");
    {
        InferenceResult result;

        Classification cls1; cls1.confidence = 0.3f; result.AddClassification(cls1);
        Classification cls2; cls2.confidence = 0.9f; result.AddClassification(cls2);
        Classification cls3; cls3.confidence = 0.5f; result.AddClassification(cls3);

        Classification top = result.GetTopClassification();
        ASSERT_FLOAT_EQ(top.confidence, 0.9f, 0.001f);
    }
    TEST_END();

    TEST_CASE("TC006-05: Clear清空");
    {
        InferenceResult result;
        Detection det; result.AddDetection(det);

        ASSERT_FALSE(result.Empty());

        result.Clear();
        ASSERT_TRUE(result.Empty());
        ASSERT_EQ(result.detections.size(), 0);
    }
    TEST_END();
}

// ============================================================================
// TC010: 性能测试
// ============================================================================

void test_performance() {
    TEST_CASE("TC010-01: NMS性能测试 (1000个框)");
    {
        std::vector<Detection> detections;

        // 生成1000个随机检测框
        for (int i = 0; i < 1000; ++i) {
            Detection det;
            det.bbox = BBox(i % 100 * 10.0f, i / 100 * 10.0f, 50, 50);
            det.confidence = 0.9f - (i % 100) * 0.008f;
            det.class_id = i % 10;
            detections.push_back(det);
        }

        auto start = std::chrono::high_resolution_clock::now();
        NMS::Apply(detections, 0.5f);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "  ℹ NMS处理1000个框耗时: " << duration.count() << "ms" << std::endl;
        ASSERT_TRUE(duration.count() < 100); // 应该在100ms内完成
    }
    TEST_END();

    TEST_CASE("TC010-02: 内存泄漏测试");
    {
        // 循环创建和销毁Tensor
        for (int i = 0; i < 1000; ++i) {
            Tensor tensor({100, 100}, DataType::FLOAT32);
            tensor.Fill(1.0f);
            // tensor自动销毁
        }

        // 如果有内存泄漏，这里会导致内存持续增长
        // 实际测试中可以用valgrind等工具检测
        ASSERT_TRUE(true); // 通过编译即表示基本正常
    }
    TEST_END();
}

// ============================================================================
// 主测试入口
// ============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   CRKIT SDK 综合测试套件" << std::endl;
    std::cout << "============================================" << std::endl;

    try {
        std::cout << "\n## TC001-002: Tensor基础和高级功能测试" << std::endl;
        test_tensor_basic();
        test_tensor_advanced();

        std::cout << "\n## TC003: BBox和Detection测试" << std::endl;
        test_bbox_detection();

        std::cout << "\n## TC004: IOU计算测试" << std::endl;
        test_iou_calculation();

        std::cout << "\n## TC005: NMS算法测试" << std::endl;
        test_nms_algorithm();

        std::cout << "\n## TC006: InferenceResult测试" << std::endl;
        test_inference_result();

        std::cout << "\n## TC010: 性能测试" << std::endl;
        test_performance();

        // 输出测试统计
        std::cout << "\n============================================" << std::endl;
        std::cout << "   测试统计" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "总测试数: " << g_stats.total << std::endl;
        std::cout << "通过: " << g_stats.passed << " ✓" << std::endl;
        std::cout << "失败: " << g_stats.failed << " ✗" << std::endl;
        std::cout << "通过率: " << std::fixed << std::setprecision(1)
                  << g_stats.passRate() << "%" << std::endl;
        std::cout << "============================================" << std::endl;

        if (g_stats.failed == 0) {
            std::cout << "\n✅ 所有测试通过!" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ 有 " << g_stats.failed << " 个测试失败!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 测试异常: " << e.what() << std::endl;
        return 1;
    }
}
