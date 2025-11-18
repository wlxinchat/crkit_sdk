#pragma once

#include "crkit/core/types.h"
#include <utility>
#include <type_traits>

namespace crkit {

/**
 * @brief Result类型 - 用于返回值或错误
 *
 * 类似于Rust的Result<T, E>或C++23的std::expected
 * 提供类型安全的错误处理机制
 *
 * @tparam T 成功值类型
 * @tparam E 错误类型，默认为Status
 */
template<typename T, typename E = Status>
class Result {
public:
    // 构造函数 - 成功值
    Result(T value) : has_value_(true) {
        new (&value_) T(std::move(value));
    }

    // 构造函数 - 错误
    Result(E error) : has_value_(false) {
        new (&error_) E(std::move(error));
    }

    // 拷贝构造
    Result(const Result& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&value_) T(other.value_);
        } else {
            new (&error_) E(other.error_);
        }
    }

    // 移动构造
    Result(Result&& other) noexcept : has_value_(other.has_value_) {
        if (has_value_) {
            new (&value_) T(std::move(other.value_));
        } else {
            new (&error_) E(std::move(other.error_));
        }
    }

    // 析构函数
    ~Result() {
        if (has_value_) {
            value_.~T();
        } else {
            error_.~E();
        }
    }

    // 拷贝赋值
    Result& operator=(const Result& other) {
        if (this != &other) {
            this->~Result();
            new (this) Result(other);
        }
        return *this;
    }

    // 移动赋值
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            this->~Result();
            new (this) Result(std::move(other));
        }
        return *this;
    }

    /**
     * @brief 检查是否包含值
     * @return true 如果包含成功值
     */
    bool IsOk() const { return has_value_; }
    bool IsError() const { return !has_value_; }

    /**
     * @brief 获取成功值（引用）
     * @warning 必须先检查IsOk()，否则行为未定义
     */
    const T& Value() const & {
        return value_;
    }

    T& Value() & {
        return value_;
    }

    T&& Value() && {
        return std::move(value_);
    }

    /**
     * @brief 获取错误（引用）
     * @warning 必须先检查IsError()，否则行为未定义
     */
    const E& Error() const & {
        return error_;
    }

    E& Error() & {
        return error_;
    }

    E&& Error() && {
        return std::move(error_);
    }

    /**
     * @brief 获取值或默认值
     * @param default_val 如果是错误则返回的默认值
     * @return 成功值或默认值
     */
    T ValueOr(T default_val) const & {
        return has_value_ ? value_ : std::move(default_val);
    }

    T ValueOr(T default_val) && {
        return has_value_ ? std::move(value_) : std::move(default_val);
    }

    /**
     * @brief 转换值（map操作）
     * @tparam F 转换函数类型
     * @param f 转换函数 T -> U
     * @return Result<U, E>
     */
    template<typename F>
    auto Map(F&& f) const & -> Result<decltype(f(value_)), E> {
        using U = decltype(f(value_));
        if (has_value_) {
            return Result<U, E>(f(value_));
        } else {
            return Result<U, E>(error_);
        }
    }

    template<typename F>
    auto Map(F&& f) && -> Result<decltype(f(std::move(value_))), E> {
        using U = decltype(f(std::move(value_)));
        if (has_value_) {
            return Result<U, E>(f(std::move(value_)));
        } else {
            return Result<U, E>(std::move(error_));
        }
    }

    /**
     * @brief 链式调用（flatMap/and_then操作）
     * @tparam F 转换函数类型
     * @param f 转换函数 T -> Result<U, E>
     * @return Result<U, E>
     */
    template<typename F>
    auto AndThen(F&& f) const & -> decltype(f(value_)) {
        if (has_value_) {
            return f(value_);
        } else {
            return decltype(f(value_))(error_);
        }
    }

    template<typename F>
    auto AndThen(F&& f) && -> decltype(f(std::move(value_))) {
        if (has_value_) {
            return f(std::move(value_));
        } else {
            return decltype(f(std::move(value_)))(std::move(error_));
        }
    }

    /**
     * @brief 转换错误（mapError操作）
     * @tparam F 转换函数类型
     * @param f 转换函数 E -> E2
     * @return Result<T, E2>
     */
    template<typename F>
    auto MapError(F&& f) const & -> Result<T, decltype(f(error_))> {
        using E2 = decltype(f(error_));
        if (has_value_) {
            return Result<T, E2>(value_);
        } else {
            return Result<T, E2>(f(error_));
        }
    }

    template<typename F>
    auto MapError(F&& f) && -> Result<T, decltype(f(std::move(error_)))> {
        using E2 = decltype(f(std::move(error_)));
        if (has_value_) {
            return Result<T, E2>(std::move(value_));
        } else {
            return Result<T, E2>(f(std::move(error_)));
        }
    }

    // 支持bool转换
    explicit operator bool() const { return has_value_; }

private:
    union {
        T value_;
        E error_;
    };
    bool has_value_;
};

/**
 * @brief 创建成功Result
 */
template<typename T>
Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

/**
 * @brief 创建错误Result
 */
template<typename T, typename E>
Result<T, E> Err(E error) {
    return Result<T, E>(std::move(error));
}

} // namespace crkit
