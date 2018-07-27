#include <algorithm>
#include "hmm.hpp"
#include "math_utils.hpp"

template<typename dtype>
HMM<dtype>::HMM() {

}

template<typename dtype>
HMM<dtype>::~HMM() {
    for (Distribution<dtype> *state : states) {
        delete state;
    }
}

template<typename dtype>
void HMM<dtype>::init() {

    const uint M = states.size();

    transition = std::log(1.0 / M) * MatrixX<dtype>::Ones(M, M);
    init_prob = std::log(1.0 / M) * VectorX<dtype>::Ones(M);

}

template<typename dtype>
void HMM<dtype>::forward_backward(const Sequence<dtype> &seq) {

    const uint M = states.size();
    const uint T = seq.rows;
    const uint D = seq.cols;

    MatrixX<dtype> alpha(M, T);
    for (uint j = 0; j < M; j++) {
        alpha(j, 0) = states[j]->logp(seq.get_row(0), D) + init_prob[j];
    }

    for (uint t = 0; t < T-1; t++) {

        for (uint j = 0; j < M; j++) {

            if (seq.labels[t] != -1) {

                uint i = seq.labels[t];

                alpha(j, t+1) = alpha(i, t) + transition(i, j) + states[j]->logp(seq.get_row(t+1), D);

            } else {
                std::vector<dtype> terms(M);

                for (uint i = 0; i < M; i++) {

                    // auto _a = alpha(i, t);
                    // auto _b = transition(i, j);
                    // auto _c = states[j]->logp(seq.get_row(t+1), D);
                    // std::cout << _a << " " << _b << " " << _c << std::endl;

                    terms[i] = alpha(i, t) + transition(i, j) + states[j]->logp(seq.get_row(t+1), D);

                }

                alpha(j, t+1) = log_sum_exp(terms.data(), terms.size());

                // for (auto o : terms) {
                //     std::cout << o << " ";
                // }
                // std::cout << std::endl;


            }


        }

    }

    MatrixX<dtype> beta(M, T);
    for (uint j = 0; j < M; j++) {
        beta(j, T-1) = 0.0;
    }
    for (uint t = T-1; t > 0; t--) {

        for (uint i = 0; i < M; i++) {

            if (seq.labels[t] != -1) {

                uint j = seq.labels[t];

                beta(i, t-1) = beta(j, t) + transition(i, j) + states[j]->logp(seq.get_row(t), D);

            } else {

                std::vector<dtype> terms(M);

                for (uint j = 0; j < M; j++) {

                    terms[j] = beta(j, t) + transition(i, j) + states[j]->logp(seq.get_row(t), D);

                }

                beta(i, t-1) = log_sum_exp(terms.data(), terms.size());
            }
        }

    }

    std::vector<dtype> terms(M);
    for (uint i = 0; i < M; i++) {
        terms[i] = alpha(i, 0) + beta(i, 0);
    }
    float p_obs = log_sum_exp(terms.data(), terms.size());

    // TODO(RL) Try to compute gamma directly !
    MatrixX<dtype> gamma(M, T);
    for (uint t = 0; t < T; t++) {

        for (uint i = 0; i < M; i++) {

            if (seq.labels[t] != -1 && (uint) seq.labels[t] != i) {
                gamma(i, t) = -INFINITY;
                continue;
            }

            gamma(i, t) = alpha(i, t) + beta(i, t) - p_obs;
        }
    }

    MatrixX<dtype> xi(M * M, T);
    for (uint t = 0; t < T - 1; t++) {

        for (uint i = 0; i < M; i++) {

            for (uint j = 0; j < M; j++) {

                uint id = i * M + j;

                if (seq.labels[t] != -1 && (uint) seq.labels[t] != i) {
                    xi(id, t) = -INFINITY;
                }

                if (seq.labels[t+1] != -1 && (uint) seq.labels[t+1] != j) {
                    xi(id, t) = -INFINITY;
                }

                xi(id, t) = alpha(i, t) + states[j]->logp(seq.get_row(t+1), D) + beta(j, t+1) + transition(i, j) - p_obs;

            }

        }

    }

    // Update transition matrix
    for (uint i = 0; i < M; i++) {
        for (uint j = 0; j < M; j++) {

            std::vector<dtype> v;
            v.reserve(T-1);
            for (uint t = 0; t < T - 1; t++) {
                uint id = i * M + j;
                if (xi(id, t) != -INFINITY) {
                    v.push_back(xi(id, t));
                }
            }
            transition(i, j) = log_sum_exp(v.data(), v.size());
        }
    }

    // TODO(RL) Screw optimality
    for (uint i = 0; i < M; i++) {
        std::vector<dtype> v;
        for (uint k = 0; k < M; k++) {
            v.push_back(transition(i, k));
        }
        dtype w = log_sum_exp(v.data(), v.size());
        for (uint j = 0; j < M; j++) {
            transition(i, j) -= w;
        }
    }

    for (uint i = 0; i < M; i++) {

        std::vector<dtype> v(T);
        for (uint t = 0; t < T; t++) {
            v[t] = gamma(i, t);
        }

        states[i]->update_params(seq.data, v.data(), seq.rows);

    }



}


template<typename dtype>
void HMM<dtype>::fit(const Sequence<dtype> *data, uint len) {

    if (len > 1) {
        std::cout << "Not supported yet !" << std::endl;
    }

    init();

    const uint max_iters = 100;

    for (uint i = 0; i < max_iters; i++) {
        std::cout << "iteration " << i << std::endl;
        forward_backward(data[0]);
    }

}

template<typename dtype>
uint HMM<dtype>::add_state(Distribution<dtype> *distribution) {
    states.push_back(distribution);
    return states.size() - 1;
}


template class HMM<float>;
template class HMM<double>;
