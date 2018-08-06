#pragma once

#define DEBUG
#ifdef DEBUG

#define CHECK_LOG_PROB(p)                                       \
    do {                                                        \
        if (p > 1e-6) {                                         \
            std::cout << "ERR: line " << __LINE__               \
                      << " log prob is " << p << std::endl;     \
            exit(-1);                                           \
        }                                                       \
    } while (false);


#else // NDEBUG

#define CHECK_LOG_PROB(p)

#endif
