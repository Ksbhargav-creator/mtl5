#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>

#include <iostream>

namespace mtl::sparse::instrumentation {

class KernelStatistics {
private:
    std::vector<std::size_t> samples;
public:
    struct StatisticSummary {
        std::size_t count;
        double mean;
        double median;
        double maximum;
    };

    void record(std::size_t length) {
        samples.push_back(length);
    }

    void reset() {
        samples.clear();
    }

    const std::vector<std::size_t>& values() const {
        return samples;
    }

    StatisticSummary summary() const {
        if (samples.empty())
            return {};

        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());

        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

        double median;
        if (sorted.size() % 2 == 0)
            median = (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
        else
            median = sorted[sorted.size()/2];

        return {
            sorted.size(),
            sum / sorted.size(),
            static_cast<double>(median),
            static_cast<double>(sorted.back())
        };
    }
};

// Global inline instances for solver instrumentation (C++17 inline variables)
inline static KernelStatistics forward_stats;
inline static KernelStatistics backward_stats;
inline static KernelStatistics schur_stats;


inline void print_stats(const char* name,
                        const KernelStatistics& stats)
{
    auto s = stats.summary();

    std::cout
        << name << '\n'
        << " count   = " << s.count << '\n'
        << " mean    = " << s.mean << '\n'
        << " median  = " << s.median << '\n'
        << " max     = " << s.maximum << "\n\n";
}

} // namespace mtl::sparse::instrumentation
