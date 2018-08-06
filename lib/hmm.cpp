#include <algorithm>
#include "hmm.hpp"
#include "math_utils.hpp"
#include "debug.hpp"

#include "vector.hpp"

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

    MatrixX<dtype> logp_states(M, T);
    for (uint t = 0; t < T; t++) {
        for (uint i = 0; i < M; i++) {
            logp_states(i, t) = states[i]->logp(seq.get_row(t), D);
        }
    }

    MatrixX<dtype> alpha(M, T);
    for (uint j = 0; j < M; j++) {
        alpha(j, 0) = init_prob[j] + logp_states(j, 0);
        CHECK_LOG_PROB(alpha(j, 0))
    }

    for (uint t = 0; t < T-1; t++) {

        for (uint j = 0; j < M; j++) {

            if (seq.labels[t] != -1) {

                uint i = seq.labels[t];

                alpha(j, t+1) = alpha(i, t) + transition(i, j) + logp_states(j, t+1);

            } else {

                std::vector<dtype> terms(M);

                for (uint i = 0; i < M; i++) {

                    // auto _a = alpha(i, t);
                    // auto _b = transition(i, j);
                    // auto _c = states[j]->logp(seq.get_row(t+1), D);
                    // std::cout << _a << " " << _b << " " << _c << std::endl;

                    terms[i] = alpha(i, t) + transition(i, j) + logp_states(j, t+1);

                }

                alpha(j, t+1) = log_sum_exp(terms.data(), terms.size());
                CHECK_LOG_PROB(alpha(j, t+1))

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

                beta(i, t-1) = beta(j, t) + transition(i, j) + logp_states(j, t);

            } else {

                std::vector<dtype> terms(M);

                for (uint j = 0; j < M; j++) {

                    terms[j] = beta(j, t) + transition(i, j) + logp_states(j, t);

                }

                beta(i, t-1) = log_sum_exp(terms.data(), terms.size());
                CHECK_LOG_PROB(beta(i, t-1))
            }
        }

    }

    std::vector<dtype> p_obs(T);
    for (uint t = 0; t < T; t++) {
        std::vector<dtype> terms(M);
        for (uint i = 0; i < M; i++) {
            terms[i] = alpha(i, t) + beta(i, t);
        }
        p_obs[t] = log_sum_exp(terms.data(), terms.size());
    }

    MatrixX<dtype> gamma(M, T);
    for (uint t = 0; t < T; t++) {

        for (uint i = 0; i < M; i++) {

            if (seq.labels[t] != -1 && (uint) seq.labels[t] != i) {
                gamma(i, t) = -INFINITY;
                continue;
            }

            gamma(i, t) = alpha(i, t) + beta(i, t) - p_obs[t];
            CHECK_LOG_PROB(gamma(i, t))
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

                xi(id, t) = alpha(i, t) + logp_states(j, t+1) + beta(j, t+1) + transition(i, j) - p_obs[t];
                CHECK_LOG_PROB(xi(id, t))
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
vector<uint> HMM<dtype>::viterbi_sequence(const Sequence<dtype> &seq) {

    uint M = states.size();
    uint T = seq.rows;
    uint D = seq.cols;

    MatrixX<dtype> logp_states(M, T);
    for (uint t = 0; t < T; t++) {
        for (uint i = 0; i < M; i++) {
            logp_states(i, t) = states[i]->logp(seq.get_row(t), D);
        }
    }

    VectorX<dtype> V(M);
    for (uint i = 0; i < M; i++) {
        V[i] = init_prob[i] + logp_states(i, 0);
    }

    MatrixX<uint> pred(M, T-1);

    VectorX<dtype> W(M);
    for (uint t = 0; t < T-1; t++) {

        for (uint i = 0; i < M; i++) {
            dtype best = std::numeric_limits<dtype>::lowest();
            uint best_id = 0;
            for (uint k = 0; k < M; k++) {
                // TODO(RL) compute states[i] outside
                dtype cur = transition(k, i) + V[k] + logp_states(i, t+1);
                if (cur > best) {
                    best = cur;
                    best_id = k;
                }
            }
            W[i] = best;
            pred(i, t) = best_id;
        }
        // TODO(RL) Swap ?
        V = W;
    }

    uint id = 0;
    for (uint i = 0; i < M; i++) {
        if (V[i] > V[id]) {
            id = i;
        }
    }

    vector<uint> result(T);
    result[T - 1] = id;

    for (uint t = T-1; t > 0; t--) {
        id = pred(id, t-1);
        result[t-1] = id;
    }

    return result;

}

template<typename dtype>
vector< vector<uint> > HMM<dtype>::viterbi(const Sequence<dtype> *data, uint len) {

    vector< vector<uint> > result(len);

    for (uint i = 0; i < len; i++) {
        result[i] = viterbi_sequence(data[i]);
    }

    return result;

}

template<typename dtype>
void HMM<dtype>::fit(const Sequence<dtype> *data, uint len, uint max_iters) {

    if (len > 1) {
        std::cout << "Not supported yet !" << std::endl;
    }

    init();

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
