#pragma once

#include "common.hpp"
#include "distribution.hpp"
#include "vector.hpp"
#include "ndarray.hpp"

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
    MatrixXR<dtype> transition;
    VectorX<dtype> init_prob;

public:
    HMM();
    ~HMM();

    void init();

    vector<uint> viterbi_sequence(const Sequence<dtype> &seq);
    vector< vector<uint> > viterbi(const Sequence<dtype> *data, uint len);
    void fit(const Sequence<dtype> *data, uint len, dtype eps = 1e-6, uint max_iters = 1000);
    uint add_state(Distribution<dtype> *distribution);

private:
    inline dtype forward_backward(const Sequence<dtype> &seq,
                                  ndarray<dtype, 2> &alpha, ndarray<dtype, 2> &beta,
                                  ndarray<dtype, 2> &gamma, ndarray<dtype, 3> &xi);
    void viterbi_iter(const Sequence<dtype> &seq, uint* result);

    MatrixXR<dtype> update_transition;


    void reset_transition_update();

    void update_transition_params(const ndarray<dtype, 3> &xi);

    void apply_transition_update();


};
