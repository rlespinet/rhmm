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

    MatrixXR<dtype> constraints;
    VectorX<dtype> constraints_pleft;
    VectorX<uint>  constraints_count;

public:
    HMM();
    ~HMM();


    vector<uint> viterbi_sequence(const Sequence<dtype> &seq);
    vector< vector<uint> > viterbi(const Sequence<dtype> *data, uint len);
    bool init_fit();
    bool fit(const Sequence<dtype> *data, uint len, dtype eps = 1e-6, uint max_iters = 1000);
    uint add_state(Distribution<dtype> *distribution);
    bool set_transition_constraints(const dtype *transition, uint M);

private:
    inline dtype forward_backward(const Sequence<dtype> &seq,
                                  MatrixX<dtype> &alpha, MatrixX<dtype> &beta,
                                  MatrixX<dtype> &gamma, MatrixX<dtype> &xi);
    void viterbi_iter(const Sequence<dtype> &seq, uint* result);

    MatrixXR<dtype> update_transition;


    void reset_transition_update();

    void update_transition_params(const MatrixX<dtype> &xi, uint T);

    void apply_transition_update();

    VectorX<dtype> update_init_prob;

    void reset_init_prob_update();

    void update_init_prob_params(const MatrixX<dtype> &gamma);

    void apply_init_prob_update();

};
