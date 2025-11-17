#pragma once

#include "crkit/core/types.h"
#include <memory>
#include <cstring>

namespace crkit {

// 张量类 - 用于存储输入输出数据
class Tensor {
public:
    Tensor() : data_(nullptr), size_(0), dtype_(DataType::FLOAT32) {}

    // 构造函数
    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype) {
        size_ = ShapeSize(shape) * GetDataTypeSize(dtype);
        data_ = std::shared_ptr<void>(malloc(size_), free);
    }

    // 从外部数据构造（零拷贝）
    Tensor(void* data, const Shape& shape, DataType dtype, bool copy = false)
        : shape_(shape), dtype_(dtype) {
        size_ = ShapeSize(shape) * GetDataTypeSize(dtype);
        if (copy) {
            data_ = std::shared_ptr<void>(malloc(size_), free);
            std::memcpy(data_.get(), data, size_);
        } else {
            // 不拷贝，直接使用外部内存（需要确保外部内存生命周期）
            data_ = std::shared_ptr<void>(data, [](void*) {});
        }
    }

    // 获取数据指针
    void* Data() { return data_.get(); }
    const void* Data() const { return data_.get(); }

    template<typename T>
    T* Data() { return static_cast<T*>(data_.get()); }

    template<typename T>
    const T* Data() const { return static_cast<const T*>(data_.get()); }

    // 获取形状
    const Shape& GetShape() const { return shape_; }

    // 获取维度
    size_t Rank() const { return shape_.size(); }

    // 获取某个维度的大小
    int64_t Dim(size_t idx) const {
        return idx < shape_.size() ? shape_[idx] : -1;
    }

    // 获取元素总数
    int64_t NumElements() const { return ShapeSize(shape_); }

    // 获取数据类型
    DataType GetDataType() const { return dtype_; }

    // 获取字节大小
    size_t ByteSize() const { return size_; }

    // 重塑形状（不改变数据）
    bool Reshape(const Shape& new_shape) {
        if (ShapeSize(new_shape) != NumElements()) {
            return false;
        }
        shape_ = new_shape;
        return true;
    }

    // 克隆张量
    Tensor Clone() const {
        Tensor new_tensor(shape_, dtype_);
        std::memcpy(new_tensor.Data(), Data(), size_);
        return new_tensor;
    }

    // 清空数据
    void Clear() {
        data_.reset();
        shape_.clear();
        size_ = 0;
    }

    // 判断是否为空
    bool Empty() const { return data_ == nullptr || size_ == 0; }

    // 填充数据
    template<typename T>
    void Fill(T value) {
        if (dtype_ != DataType::FLOAT32 && dtype_ != DataType::INT32) {
            return; // 仅支持部分类型
        }
        T* ptr = Data<T>();
        for (int64_t i = 0; i < NumElements(); ++i) {
            ptr[i] = value;
        }
    }

private:
    std::shared_ptr<void> data_;  // 数据指针（使用智能指针自动管理内存）
    Shape shape_;                  // 张量形状
    size_t size_;                  // 字节大小
    DataType dtype_;               // 数据类型
};

// 便捷的张量创建函数
inline Tensor MakeTensor(const Shape& shape, DataType dtype = DataType::FLOAT32) {
    return Tensor(shape, dtype);
}

// 从原始数据创建张量
template<typename T>
Tensor MakeTensor(T* data, const Shape& shape, bool copy = true) {
    DataType dtype = DataType::FLOAT32;
    if (std::is_same<T, float>::value) dtype = DataType::FLOAT32;
    else if (std::is_same<T, int>::value) dtype = DataType::INT32;
    else if (std::is_same<T, uint8_t>::value) dtype = DataType::UINT8;

    return Tensor(data, shape, dtype, copy);
}

} // namespace crkit
