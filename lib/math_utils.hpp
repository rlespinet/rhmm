#pragma once

#include <cmath>

// Use the log sum exp trick to avoid underflows
template<typename dtype>
dtype log_sum_exp(const dtype *data, uint len) {

    dtype max = *std::max_element(data, data + len);
    if (max == -INFINITY) {
        return -INFINITY;
    }

    // TODO(RL) SIMD ? !
    dtype result = 0.0;
    for (uint i = 0; i < len; i++) {
        result += std::exp(data[i] - max);
    }

    return std::log(result) + max;
}
