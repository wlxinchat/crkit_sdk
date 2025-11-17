#include "crkit/data/tensor.h"
#include <cassert>
#include <iostream>
#include <cstring>

using namespace crkit;

void test_tensor_creation() {
    std::cout << "测试: Tensor创建..." << std::endl;

    // 测试空Tensor
    Tensor empty_tensor;
    assert(empty_tensor.Empty());
    assert(empty_tensor.NumElements() == 0);

    // 测试创建指定形状的Tensor
    Shape shape = {1, 3, 640, 640};
    Tensor tensor(shape, DataType::FLOAT32);

    assert(!tensor.Empty());
    assert(tensor.GetShape() == shape);
    assert(tensor.Rank() == 4);
    assert(tensor.NumElements() == 1 * 3 * 640 * 640);
    assert(tensor.GetDataType() == DataType::FLOAT32);

    std::cout << "  ✓ Tensor创建测试通过" << std::endl;
}

void test_tensor_data_access() {
    std::cout << "测试: Tensor数据访问..." << std::endl;

    Shape shape = {2, 3};
    Tensor tensor(shape, DataType::FLOAT32);

    // 写入数据
    float* data = tensor.Data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 读取数据
    const float* const_data = tensor.Data<const float>();
    for (int i = 0; i < 6; ++i) {
        assert(const_data[i] == static_cast<float>(i));
    }

    std::cout << "  ✓ Tensor数据访问测试通过" << std::endl;
}

void test_tensor_reshape() {
    std::cout << "测试: Tensor重塑..." << std::endl;

    Shape shape = {2, 3, 4};
    Tensor tensor(shape, DataType::FLOAT32);

    // 重塑为相同元素数的形状
    Shape new_shape = {1, 24};
    bool success = tensor.Reshape(new_shape);

    assert(success);
    assert(tensor.GetShape() == new_shape);
    assert(tensor.NumElements() == 24);

    // 尝试重塑为不同元素数的形状（应该失败）
    Shape invalid_shape = {1, 25};
    success = tensor.Reshape(invalid_shape);
    assert(!success);

    std::cout << "  ✓ Tensor重塑测试通过" << std::endl;
}

void test_tensor_clone() {
    std::cout << "测试: Tensor克隆..." << std::endl;

    Shape shape = {2, 2};
    Tensor tensor(shape, DataType::FLOAT32);

    // 写入数据
    float* data = tensor.Data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    // 克隆
    Tensor cloned = tensor.Clone();

    assert(cloned.GetShape() == tensor.GetShape());
    assert(cloned.NumElements() == tensor.NumElements());

    // 验证数据
    const float* cloned_data = cloned.Data<const float>();
    assert(cloned_data[0] == 1.0f);
    assert(cloned_data[1] == 2.0f);
    assert(cloned_data[2] == 3.0f);
    assert(cloned_data[3] == 4.0f);

    // 修改原Tensor不应影响克隆
    data[0] = 99.0f;
    assert(cloned_data[0] == 1.0f);

    std::cout << "  ✓ Tensor克隆测试通过" << std::endl;
}

void test_tensor_fill() {
    std::cout << "测试: Tensor填充..." << std::endl;

    Shape shape = {3, 3};
    Tensor tensor(shape, DataType::FLOAT32);

    tensor.Fill(5.0f);

    const float* data = tensor.Data<const float>();
    for (int i = 0; i < 9; ++i) {
        assert(data[i] == 5.0f);
    }

    std::cout << "  ✓ Tensor填充测试通过" << std::endl;
}

int main() {
    std::cout << "=== Tensor单元测试 ===" << std::endl << std::endl;

    try {
        test_tensor_creation();
        test_tensor_data_access();
        test_tensor_reshape();
        test_tensor_clone();
        test_tensor_fill();

        std::cout << "\n所有测试通过! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n测试失败: " << e.what() << std::endl;
        return 1;
    }
}
