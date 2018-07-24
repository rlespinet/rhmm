#pragma once

#include "common.hpp"
#include "distribution.hpp"

template<typename dtype>
struct Sequence {
    dtype *data;
    int *labels;
    uint rows;
    uint cols;

    Sequence(dtype* data, int* labels, uint rows, uint cols)
        : data(data)
        , labels(labels)
        , rows(rows)
        , cols(cols) {
    }

    dtype *get_row(uint t) {
        return &data[t * cols];
    }

    const dtype *get_row(uint t) const {
        return &data[t * cols];
    }


};

template<typename dtype>
struct HMM {

    std::vector< Distribution<dtype> *> states;
    MatrixX<dtype> transition;
    VectorX<dtype> init_prob;

public:
    HMM();
    ~HMM();

    void init();
    void forward_backward(const Sequence<dtype> &seq);
    void fit(const Sequence<dtype> *data, uint len);
    uint add_state(Distribution<dtype> *distribution);

};
