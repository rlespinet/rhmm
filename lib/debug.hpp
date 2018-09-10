#pragma once

#define DEBUG

#ifdef DEBUG
#undef NDEBUG

#include <fstream>

namespace debug {

    template<typename T>
    void dump_mat(T &mat, const char* path) {
        std::ofstream f(path, std::ios_base::out);
        for (uint i = 0; i < mat.rows(); i++) {
            for (uint j = 0; j < mat.cols(); j++) {
                f << mat(i, j) << ", ";
            }
            f << std::endl;
        }
    }

}

#endif

#include <cmath>
#include <cassert>

template<typename T>
inline bool assert_not_nan(T t) {
    assert(!std::isnan(t));
}

template<typename T>
inline bool assert_negative_smooth(T t, T eps=1e-6) {
    assert(t < eps);
}

template<typename T>
inline bool assert_positive_smooth(T t, T eps=1e-6) {
    assert(t > -eps);
}
