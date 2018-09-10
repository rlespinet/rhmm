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
}

template<typename dtype>
void HMM<dtype>::init() {

    const uint M = states.size();

    transition = std::log(1.0 / M) * MatrixX<dtype>::Ones(M, M);
    init_prob = std::log(1.0 / M) * VectorX<dtype>::Ones(M);

}

template<typename dtype>
inline dtype HMM<dtype>::forward_backward(const Sequence<dtype> &seq,
                                          MatrixX<dtype> &alpha, MatrixX<dtype> &beta,
                                          MatrixX<dtype> &gamma, MatrixX<dtype> &xi) {

    const uint M = states.size();
    const uint T = seq.rows;
    const uint D = seq.cols;

    MatrixX<dtype> logp_states(M, T);
    for (uint t = 0; t < T; t++) {
        for (uint i = 0; i < M; i++) {
            logp_states(i, t) = states[i]->logp(seq.get_row(t), D);
        }
    }

    for (uint j = 0; j < M; j++) {
        alpha(j, 0) = init_prob[j] + logp_states(j, 0);
        assert_not_nan(alpha(j, 0));
    }

    for (uint t = 0; t < T-1; t++) {

        for (uint j = 0; j < M; j++) {

            if (seq.labels[t] != -1) {

                uint i = seq.labels[t];

                alpha(j, t+1) = alpha(i, t) + transition(i, j) + logp_states(j, t+1);

            } else {

                std::vector<dtype> terms(M);

                for (uint i = 0; i < M; i++) {

                    terms[i] = alpha(i, t) + transition(i, j) + logp_states(j, t+1);

                }

                alpha(j, t+1) = log_sum_exp(terms.data(), terms.size());
                assert_not_nan(alpha(j, t+1));

            }

        }

    }

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
                assert_not_nan(beta(i, t-1));
            }
        }

    }

    std::vector<dtype> p_obs(T);
    for (uint t = 0; t < T; t++) {

        if (seq.labels[t] != -1) {

            uint i = seq.labels[t];

            p_obs[t] = alpha(i, t) + beta(i, t);

        } else {

            std::vector<dtype> terms(M);
            for (uint i = 0; i < M; i++) {
                terms[i] = alpha(i, t) + beta(i, t);
            }
            p_obs[t] = log_sum_exp(terms.data(), terms.size());

        }

        assert_negative_smooth(p_obs[t]);
    }

    // TODO(RL) we should warn the user if p_obs is -inf
    // which probably means that data with these labels
    // cannot occur with these constraints. at this point
    // gamma and xi don't make sense, we should also drop
    // the sequence

    for (uint t = 0; t < T; t++) {

        for (uint i = 0; i < M; i++) {

            if (seq.labels[t] != -1 && (uint) seq.labels[t] != i) {
                gamma(i, t) = -INFINITY;
                continue;
            }

            gamma(i, t) = alpha(i, t) + beta(i, t) - p_obs[t];
            assert_negative_smooth(gamma(i, t));
        }
    }

    for (uint t = 0; t < T - 1; t++) {

        for (uint i = 0; i < M; i++) {

            for (uint j = 0; j < M; j++) {

                uint id = i * M + j;

                if (seq.labels[t] != -1 && (uint) seq.labels[t] != i) {
                    xi(id, t) = -INFINITY;
                    continue;
                }

                if (seq.labels[t+1] != -1 && (uint) seq.labels[t+1] != j) {
                    xi(id, t) = -INFINITY;
                    continue;
                }

                xi(id, t) = alpha(i, t) + logp_states(j, t+1) + beta(j, t+1) + transition(i, j) - p_obs[t];
                assert_negative_smooth(xi(id, t));
            }

        }

    }

    return p_obs[T-1];

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
            if (seq.labels[t] != -1) {

                uint k = seq.labels[t];

                W[i] = transition(k, i) + V[k] + logp_states(i, t+1);
                pred(i, t) = k;

            } else {

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
        }
        // TODO(RL) Swap ?
        V = W;
    }

    uint id;
    if (seq.labels[T - 1] != -1) {
        id = seq.labels[T - 1];
    } else {
        id = 0;
        for (uint i = 0; i < M; i++) {
            if (V[i] > V[id]) {
                id = i;
            }
        }
        // result[T - 1] = std::distance(V, std::max_element(V, V + M))
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
void HMM<dtype>::reset_transition_update() {

    const uint M = states.size();

    update_transition = std::numeric_limits<dtype>::lowest() * MatrixX<dtype>::Ones(M, M);

}

template<typename dtype>
void HMM<dtype>::update_transition_params(const MatrixX<dtype> &xi, uint T) {

    const uint M = states.size();

    if (T < 2) {
        return;
    }

    // Update transition matrix
    for (uint i = 0; i < M; i++) {
        for (uint j = 0; j < M; j++) {

            uint id = i * M + j;

            std::vector<dtype> v;
            v.reserve(T-1);
            for (uint t = 0; t < T - 1; t++) {
                if (xi(id, t) != -INFINITY) {
                    v.push_back(xi(id, t));
                }
            }
            update_transition(i, j) = log_sum_exp(update_transition(i, j),
                                                  log_sum_exp(v.data(), v.size()));
        }
    }

}

template<typename dtype>
void HMM<dtype>::apply_transition_update() {

    const uint M = update_transition.rows();

    // TODO(RL) Screw optimality
    for (uint i = 0; i < M; i++) {
        std::vector<dtype> v;
        for (uint k = 0; k < M; k++) {
            v.push_back(update_transition(i, k));
        }
        dtype w = log_sum_exp(v.data(), v.size());
        for (uint j = 0; j < M; j++) {
            update_transition(i, j) -= w;
        }
    }

    transition = update_transition;

}


template<typename dtype>
void HMM<dtype>::reset_init_prob_update() {

    const uint M = states.size();

    update_init_prob = std::numeric_limits<dtype>::lowest() * VectorX<dtype>::Ones(M);

}

template<typename dtype>
void HMM<dtype>::update_init_prob_params(const MatrixX<dtype> &gamma) {

    const uint M = states.size();

    for (uint i = 0; i < M; i++) {
        update_init_prob[i] = log_sum_exp(update_init_prob[i], gamma(i, 0));
    }


}

template<typename dtype>
void HMM<dtype>::apply_init_prob_update() {

    const uint M = update_init_prob.size();

    dtype w = log_sum_exp(update_init_prob.data(), update_init_prob.size());
    for (uint i = 0; i < M; i++) {
        update_init_prob[i] -= w;
    }

    init_prob = update_init_prob;

}


template<typename dtype>
void HMM<dtype>::fit(const Sequence<dtype> *data, uint len, dtype eps, uint max_iters) {

    uint max_rows = std::numeric_limits<uint>::lowest();
    for (uint i = 0; i < len; i++) {
        max_rows = std::max(max_rows, data[i].rows);
    }

    const uint M = states.size();

    init();

    MatrixX<dtype> alpha(M, max_rows);
    MatrixX<dtype> beta(M, max_rows);
    MatrixX<dtype> gamma(M, max_rows);
    MatrixX<dtype> xi(M * M, max_rows);

    dtype likelihood = std::numeric_limits<dtype>::lowest();
    bool converged = false;

    for (uint n = 0; n < max_iters && !converged; n++) {

        std::cout << "iteration " << n << std::endl;

        reset_transition_update();

        reset_init_prob_update();

        for (uint i = 0; i < M; i++) {
            states[i]->reset_update();
        }

        dtype new_likelihood = 0.0;
        for (uint k = 0; k < len; k++) {

            const Sequence<dtype> &seq = data[k];

            const uint D = seq.cols;
            const uint T = seq.rows;

            new_likelihood += forward_backward(seq, alpha, beta, gamma, xi);

            update_transition_params(xi, T);

            update_init_prob_params(gamma);

            for (uint i = 0; i < M; i++) {

                std::vector<dtype> v(T);
                for (uint t = 0; t < T; t++) {
                    v[t] = gamma(i, t);
                }

                states[i]->update_params(seq.data, v.data(), seq.rows);

            }

        }

        for (uint i = 0; i < M; i++) {
            states[i]->apply_update();
        }

        apply_transition_update();

        apply_init_prob_update();

        if (std::abs(new_likelihood - likelihood) < eps) {
            break;
        }

        likelihood = new_likelihood;
        std::cout << "likelihood : " << likelihood << std::endl;

    }

}

template<typename dtype>
uint HMM<dtype>::add_state(Distribution<dtype> *distribution) {
    states.push_back(distribution);
    return states.size() - 1;
}


template class HMM<float>;
template class HMM<double>;
