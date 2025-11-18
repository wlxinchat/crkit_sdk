#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>

namespace crkit {

/**
 * @brief 性能指标收集器
 *
 * 提供推理性能统计功能，包括：
 * - 延迟统计（平均、P50、P95、P99）
 * - 吞吐量统计
 * - 成功/失败计数
 * - Prometheus格式导出
 */
class PerformanceMetrics {
public:
    /**
     * @brief 推理性能指标
     */
    struct InferenceMetrics {
        double avg_latency_ms = 0.0;     ///< 平均延迟（毫秒）
        double min_latency_ms = 0.0;     ///< 最小延迟（毫秒）
        double max_latency_ms = 0.0;     ///< 最大延迟（毫秒）
        double p50_latency_ms = 0.0;     ///< P50延迟（毫秒）
        double p95_latency_ms = 0.0;     ///< P95延迟（毫秒）
        double p99_latency_ms = 0.0;     ///< P99延迟（毫秒）
        uint64_t total_requests = 0;     ///< 总请求数
        uint64_t successful_requests = 0; ///< 成功请求数
        uint64_t failed_requests = 0;    ///< 失败请求数
        double throughput_qps = 0.0;     ///< 吞吐量（QPS）
        double total_time_s = 0.0;       ///< 总运行时间（秒）
    };

    PerformanceMetrics()
        : total_requests_(0)
        , successful_requests_(0)
        , failed_requests_(0)
        , start_time_(std::chrono::steady_clock::now()) {
    }

    /**
     * @brief 记录一次推理
     * @param latency_ms 延迟（毫秒）
     * @param success 是否成功
     */
    void RecordInference(double latency_ms, bool success = true) {
        std::lock_guard<std::mutex> lock(mutex_);

        latencies_.push_back(latency_ms);
        total_requests_++;

        if (success) {
            successful_requests_++;
        } else {
            failed_requests_++;
        }
    }

    /**
     * @brief 记录批量推理
     * @param batch_size 批量大小
     * @param latency_ms 批量推理延迟（毫秒）
     * @param success 是否成功
     */
    void RecordBatchInference(int batch_size, double latency_ms, bool success = true) {
        std::lock_guard<std::mutex> lock(mutex_);

        double per_item_latency = latency_ms / batch_size;
        for (int i = 0; i < batch_size; ++i) {
            latencies_.push_back(per_item_latency);
        }

        total_requests_ += batch_size;
        if (success) {
            successful_requests_ += batch_size;
        } else {
            failed_requests_ += batch_size;
        }
    }

    /**
     * @brief 获取性能指标
     * @return InferenceMetrics 当前性能指标
     */
    InferenceMetrics GetMetrics() const {
        std::lock_guard<std::mutex> lock(mutex_);

        InferenceMetrics metrics;
        metrics.total_requests = total_requests_.load();
        metrics.successful_requests = successful_requests_.load();
        metrics.failed_requests = failed_requests_.load();

        if (latencies_.empty()) {
            return metrics;
        }

        // 计算总运行时间
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        metrics.total_time_s = duration.count() / 1000000.0;

        // 计算吞吐量
        if (metrics.total_time_s > 0) {
            metrics.throughput_qps = metrics.total_requests / metrics.total_time_s;
        }

        // 复制并排序延迟数据
        std::vector<double> sorted_latencies = latencies_;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());

        // 计算统计值
        metrics.min_latency_ms = sorted_latencies.front();
        metrics.max_latency_ms = sorted_latencies.back();

        double sum = 0.0;
        for (double lat : sorted_latencies) {
            sum += lat;
        }
        metrics.avg_latency_ms = sum / sorted_latencies.size();

        // 计算百分位数
        metrics.p50_latency_ms = Percentile(sorted_latencies, 0.50);
        metrics.p95_latency_ms = Percentile(sorted_latencies, 0.95);
        metrics.p99_latency_ms = Percentile(sorted_latencies, 0.99);

        return metrics;
    }

    /**
     * @brief 重置统计
     */
    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);

        latencies_.clear();
        total_requests_ = 0;
        successful_requests_ = 0;
        failed_requests_ = 0;
        start_time_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief 导出Prometheus格式指标
     * @param namespace_prefix 命名空间前缀
     * @return Prometheus格式字符串
     */
    std::string ExportPrometheus(const std::string& namespace_prefix = "crkit") const {
        auto metrics = GetMetrics();

        std::ostringstream oss;

        // 总请求数
        oss << "# HELP " << namespace_prefix << "_inference_requests_total Total inference requests\n";
        oss << "# TYPE " << namespace_prefix << "_inference_requests_total counter\n";
        oss << namespace_prefix << "_inference_requests_total{status=\"success\"} "
            << metrics.successful_requests << "\n";
        oss << namespace_prefix << "_inference_requests_total{status=\"failed\"} "
            << metrics.failed_requests << "\n";

        // 延迟统计
        oss << "# HELP " << namespace_prefix << "_inference_latency_ms Inference latency in milliseconds\n";
        oss << "# TYPE " << namespace_prefix << "_inference_latency_ms summary\n";
        oss << namespace_prefix << "_inference_latency_ms{quantile=\"0.5\"} "
            << std::fixed << std::setprecision(2) << metrics.p50_latency_ms << "\n";
        oss << namespace_prefix << "_inference_latency_ms{quantile=\"0.95\"} "
            << std::fixed << std::setprecision(2) << metrics.p95_latency_ms << "\n";
        oss << namespace_prefix << "_inference_latency_ms{quantile=\"0.99\"} "
            << std::fixed << std::setprecision(2) << metrics.p99_latency_ms << "\n";
        oss << namespace_prefix << "_inference_latency_ms_sum "
            << std::fixed << std::setprecision(2) << (metrics.avg_latency_ms * metrics.total_requests) << "\n";
        oss << namespace_prefix << "_inference_latency_ms_count "
            << metrics.total_requests << "\n";

        // 吞吐量
        oss << "# HELP " << namespace_prefix << "_inference_throughput_qps Inference throughput in QPS\n";
        oss << "# TYPE " << namespace_prefix << "_inference_throughput_qps gauge\n";
        oss << namespace_prefix << "_inference_throughput_qps "
            << std::fixed << std::setprecision(2) << metrics.throughput_qps << "\n";

        return oss.str();
    }

    /**
     * @brief 打印性能报告
     */
    void PrintReport() const {
        auto metrics = GetMetrics();

        std::cout << "\n========================================\n";
        std::cout << "   Performance Metrics Report\n";
        std::cout << "========================================\n";
        std::cout << "Total Requests:    " << metrics.total_requests << "\n";
        std::cout << "Successful:        " << metrics.successful_requests
                  << " (" << (metrics.total_requests > 0 ?
                     100.0 * metrics.successful_requests / metrics.total_requests : 0)
                  << "%)\n";
        std::cout << "Failed:            " << metrics.failed_requests
                  << " (" << (metrics.total_requests > 0 ?
                     100.0 * metrics.failed_requests / metrics.total_requests : 0)
                  << "%)\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Latency (ms):\n";
        std::cout << "  Average:         " << std::fixed << std::setprecision(2)
                  << metrics.avg_latency_ms << "\n";
        std::cout << "  Min:             " << std::fixed << std::setprecision(2)
                  << metrics.min_latency_ms << "\n";
        std::cout << "  Max:             " << std::fixed << std::setprecision(2)
                  << metrics.max_latency_ms << "\n";
        std::cout << "  P50:             " << std::fixed << std::setprecision(2)
                  << metrics.p50_latency_ms << "\n";
        std::cout << "  P95:             " << std::fixed << std::setprecision(2)
                  << metrics.p95_latency_ms << "\n";
        std::cout << "  P99:             " << std::fixed << std::setprecision(2)
                  << metrics.p99_latency_ms << "\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Throughput:        " << std::fixed << std::setprecision(2)
                  << metrics.throughput_qps << " QPS\n";
        std::cout << "Total Time:        " << std::fixed << std::setprecision(2)
                  << metrics.total_time_s << " s\n";
        std::cout << "========================================\n\n";
    }

private:
    /**
     * @brief 计算百分位数
     * @param sorted_data 已排序的数据
     * @param percentile 百分位（0.0-1.0）
     * @return 百分位数值
     */
    static double Percentile(const std::vector<double>& sorted_data, double percentile) {
        if (sorted_data.empty()) {
            return 0.0;
        }

        double index = percentile * (sorted_data.size() - 1);
        size_t lower_index = static_cast<size_t>(index);
        size_t upper_index = lower_index + 1;

        if (upper_index >= sorted_data.size()) {
            return sorted_data.back();
        }

        double weight = index - lower_index;
        return sorted_data[lower_index] * (1.0 - weight) +
               sorted_data[upper_index] * weight;
    }

    mutable std::mutex mutex_;
    std::vector<double> latencies_;
    std::atomic<uint64_t> total_requests_;
    std::atomic<uint64_t> successful_requests_;
    std::atomic<uint64_t> failed_requests_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace crkit
