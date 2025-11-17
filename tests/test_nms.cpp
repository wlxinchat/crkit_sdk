#include "crkit/postprocess/detection_postprocessor.h"
#include "crkit/data/inference_result.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace crkit;
using namespace crkit::postprocess;

void test_iou_computation() {
    std::cout << "测试: IOU计算..." << std::endl;

    // 完全重叠
    BBox box1(0, 0, 100, 100);
    BBox box2(0, 0, 100, 100);
    float iou = NMS::ComputeIOU(box1, box2);
    assert(std::abs(iou - 1.0f) < 0.001f);

    // 完全不重叠
    BBox box3(0, 0, 100, 100);
    BBox box4(200, 200, 100, 100);
    iou = NMS::ComputeIOU(box3, box4);
    assert(iou == 0.0f);

    // 部分重叠
    BBox box5(0, 0, 100, 100);    // 面积: 10000
    BBox box6(50, 50, 100, 100);  // 面积: 10000
    // 重叠区域: 50x50 = 2500
    // 并集: 10000 + 10000 - 2500 = 17500
    // IOU = 2500 / 17500 = 0.142857
    iou = NMS::ComputeIOU(box5, box6);
    assert(std::abs(iou - 0.142857f) < 0.001f);

    std::cout << "  ✓ IOU计算测试通过" << std::endl;
}

void test_nms_suppression() {
    std::cout << "测试: NMS抑制..." << std::endl;

    std::vector<Detection> detections;

    // 创建重叠的检测框（同类）
    Detection det1;
    det1.bbox = BBox(10, 10, 100, 100);
    det1.confidence = 0.9f;
    det1.class_id = 0;
    detections.push_back(det1);

    Detection det2;
    det2.bbox = BBox(15, 15, 100, 100);  // 高度重叠
    det2.confidence = 0.8f;  // 较低置信度
    det2.class_id = 0;
    detections.push_back(det2);

    Detection det3;
    det3.bbox = BBox(200, 200, 100, 100);  // 不重叠
    det3.confidence = 0.85f;
    det3.class_id = 0;
    detections.push_back(det3);

    // 应用NMS (IOU阈值 = 0.5)
    NMS::Apply(detections, 0.5f);

    // 应该保留2个框：det1（最高置信度）和det3（不重叠）
    assert(detections.size() == 2);
    assert(detections[0].confidence == 0.9f);   // det1
    assert(detections[1].confidence == 0.85f);  // det3

    std::cout << "  ✓ NMS抑制测试通过" << std::endl;
}

void test_nms_different_classes() {
    std::cout << "测试: NMS多类别..." << std::endl;

    std::vector<Detection> detections;

    // 重叠但不同类别的框
    Detection det1;
    det1.bbox = BBox(10, 10, 100, 100);
    det1.confidence = 0.9f;
    det1.class_id = 0;
    detections.push_back(det1);

    Detection det2;
    det2.bbox = BBox(15, 15, 100, 100);  // 高度重叠
    det2.confidence = 0.8f;
    det2.class_id = 1;  // 不同类别
    detections.push_back(det2);

    // 应用NMS
    NMS::Apply(detections, 0.5f);

    // 应该保留两个框（不同类别不互相抑制）
    assert(detections.size() == 2);

    std::cout << "  ✓ NMS多类别测试通过" << std::endl;
}

void test_nms_max_detections() {
    std::cout << "测试: NMS最大检测数..." << std::endl;

    std::vector<Detection> detections;

    // 创建10个不重叠的检测框
    for (int i = 0; i < 10; ++i) {
        Detection det;
        det.bbox = BBox(i * 150.0f, 0, 100, 100);
        det.confidence = 0.9f - i * 0.05f;  // 递减的置信度
        det.class_id = 0;
        detections.push_back(det);
    }

    // 限制最多5个检测
    NMS::Apply(detections, 0.5f, 5);

    // 应该只保留5个最高置信度的
    assert(detections.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        assert(detections[i].confidence == 0.9f - i * 0.05f);
    }

    std::cout << "  ✓ NMS最大检测数测试通过" << std::endl;
}

int main() {
    std::cout << "=== NMS单元测试 ===" << std::endl << std::endl;

    try {
        test_iou_computation();
        test_nms_suppression();
        test_nms_different_classes();
        test_nms_max_detections();

        std::cout << "\n所有测试通过! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n测试失败: " << e.what() << std::endl;
        return 1;
    }
}
