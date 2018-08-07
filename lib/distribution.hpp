#pragma once

#include "eigen_utils.hpp"
#include "math_utils.hpp"

#include <iostream>


template<typename dtype>
struct Distribution {

    virtual ~Distribution() = 0;
    virtual dtype logp(const dtype *data, uint len) = 0;
    virtual void reset_update() = 0;
    virtual void update_params(const dtype *data, const dtype *gamma, uint len) = 0;
    virtual void apply_update() = 0;

};

template<typename dtype>
Distribution<dtype>::~Distribution() {
}


template<typename dtype>
struct MultivariateGaussian : Distribution<dtype> {

    VectorX<dtype> mean;
    MatrixX<dtype> cov;

    MultivariateGaussian(const MatrixX<dtype> &mean, const MatrixX<dtype> &cov)
        : mean(mean)
        , cov(cov)
        {}

    MultivariateGaussian(MatrixX<dtype> &&mean, MatrixX<dtype> &&cov)
        : mean(mean)
        , cov(cov)
        {}

    virtual ~MultivariateGaussian() {
    }

    virtual dtype logp(const dtype *data, uint len) {
        // TODO(RL) ....
        VectorX<dtype> x(len);
        for (uint i = 0; i < len; i++) {
            x[i] = data[i] - mean[i];
        }

        dtype q = x.transpose() * (inv_cov() * x);
        dtype result = - 0.5 * len * std::log(2.0 * M_PI) - 0.5 * log_det_cov() - 0.5 * q;
        return result;
    }

    virtual void reset_update() {

        uint D = mean.size();

        update_mean = VectorX<dtype>::Zero(D);
        update_cov = MatrixX<dtype>::Zero(D, D);

        update_weight_sum = 0.0;
    }

    virtual void update_params(const dtype *data, const dtype *gamma, uint T) {
        // TODO(RL) Implement
        uint D = mean.size();


        dtype logsum_gamma = log_sum_exp(gamma, T);

        dtype *weight = new dtype[T];
        for (uint t = 0; t < T; t++) {
            weight[t] = std::exp(gamma[t] - logsum_gamma);
        }

        for (uint t = 0; t < T; t++) {
            update_weight_sum += weight[t];
        }

        VectorX<dtype> new_update_mean = update_mean;
        for (uint t = 0; t < T; t++) {
            const VectorX<dtype> data_t = Map< VectorX<dtype> >(const_cast<dtype*>(data) + D * t, D);
            new_update_mean += weight[t] * (data_t - update_mean);
        }
        new_update_mean /= update_weight_sum;

        for (uint t = 0; t < T; t++) {
            const VectorX<dtype> data_t = Map< VectorX<dtype> >(const_cast<dtype*>(data) + D * t, D);
            update_cov += (weight[t] * (data_t - new_update_mean)) * (data_t - update_mean).transpose();
        }

        update_mean = new_update_mean;

        delete[] weight;
    }

    void apply_update() {
        mean = update_mean;
        cov = update_cov / update_weight_sum;
    }

private:

    // TODO(RL) This is really inefficient, cache the results as soon as it works !!

    // MatrixX<dtype> inv_cov;
    // dtype inv_cov_det;
    // bool inv_cov_validity;

    MatrixX<dtype> inv_cov() {
        const uint D = cov.rows();
        return cov.llt().solve(MatrixX<dtype>::Identity(D, D));
    }

    dtype log_det_cov() {
        dtype result = logdet(cov);
        return result;
    }

    VectorX<dtype> update_mean;
    MatrixX<dtype> update_cov;
    dtype update_weight_sum;

};


template<typename dtype>
struct Multinomial : Distribution<dtype> {

    VectorX<dtype> log_probs;

    Multinomial(const VectorX<dtype> &probs)
        : log_probs(probs.size()) {
        // TODO(RL) ...
        for (uint i = 0; i < probs.size(); i++) {
            log_probs[i] = std::log(probs[i]);
        }
    }

    virtual ~Multinomial() {
    }

    virtual dtype logp(const dtype *data, uint len) {
        if (len != 1) {
            std::cout << "Error : Multinomial state cannot be used when there is more than one feature" << std::endl;
        }

        // TODO(RL) data should not be of type dtype
        return log_probs[(int)*data];
    }

    virtual void reset_update() {

        uint K = log_probs.size();

        update_log_probs = VectorX<dtype>::Zero(K);
    }

    virtual void update_params(const dtype *data, const dtype *gamma, uint T) {

        // TODO(RL) This is highly inefficient
        for (uint i = 0; i < update_log_probs.size(); i++) {

            std::vector<dtype> v;
            v.reserve(T);
            for (uint t = 0; t < T; t++) {
                if (data[t] == i && gamma[t] != -INFINITY) {
                    v.push_back(gamma[t]);
                }
            }
            update_log_probs[i] += log_sum_exp(v.data(), v.size());
        }

    }

    virtual void apply_update() {

        dtype sum = log_sum_exp(update_log_probs.data(), update_log_probs.size());
        for (uint i = 0; i < update_log_probs.size(); i++) {
            update_log_probs[i] -= sum;
        }

        log_probs = update_log_probs;

    }

private:

    VectorX<dtype> update_log_probs;


};
