#pragma once

#define DEBUG
#ifdef DEBUG

#include <cmath>

#define CHECK_LOG_PROB(p)                                       \
    do {                                                        \
        if (p > 1e-6 || std::isnan(p)) {                        \
            std::cout << "ERR: line " << __LINE__               \
                      << " log prob is " << p << std::endl;     \
            assert(0);                                          \
        }                                                       \
    } while (false);


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


#else // NDEBUG

#define CHECK_LOG_PROB(p)

#endif
