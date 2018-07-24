#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace Eigen;

template<typename dtype>
using MatrixX = Matrix<dtype, Dynamic, Dynamic>;

template<typename dtype>
using MatrixXR = Matrix<dtype, Dynamic, Dynamic, RowMajor>;

template<typename dtype>
using VectorX = Matrix<dtype, Dynamic, 1>;

template<typename mat>
mat get_sub_matrix(const mat &M,
                   const std::vector<uint> &ind_rows,
                   const std::vector<uint> &ind_cols) {

    mat sub_M(ind_rows.size(), ind_cols.size());
    for (uint j = 0; j < ind_cols.size(); j++) {
        for (uint i = 0; i < ind_rows.size(); i++) {
            sub_M(i, j) = M(ind_rows[i], ind_cols[j]);
        }
    }

    return sub_M;
}

template<typename mat>
mat get_sub_matrix(const mat &M,
                   const std::vector<uint> &indices) {
    return get_sub_matrix(M, indices, indices);
}

template<typename mat1, typename mat2>
void set_sub_matrix(mat1 &M,
                    const std::vector<uint> &ind_rows,
                    const std::vector<uint> &ind_cols,
                    const mat2 &sub_M) {

    for (int j = 0; j < ind_cols.size(); j++) {
        for (int i = 0; i < ind_rows.size(); i++) {
            M(ind_rows[i], ind_cols[j]) = sub_M(i, j);
        }
    }
}

template<typename mat1, typename mat2>
void set_sub_matrix(mat1 &M,
                    const std::vector<uint> &indices,
                    const mat2 &sub_M) {

    return set_sub_matrix(M, indices, indices, sub_M);
}

template<typename vec>
vec get_sub_vector(const vec &V,
                   const std::vector<uint> &indices) {

    vec sub_V(indices.size());
    for (uint i = 0; i < indices.size(); i++) {
        sub_V(i) = V(indices[i]);
    }

    return sub_V;
}


template<typename vec1, typename vec2>
void set_sub_vector(vec1 &V,
                    const std::vector<uint> &indices,
                    const vec2 &sub_V) {

    for (uint i = 0; i < indices.size(); i++) {
        V(indices[i]) = sub_V(i);
    }
}


template<typename mat1, typename mat2>
void add_sub_matrix(mat1 &M,
                    const std::vector<uint> &ind_rows,
                    const std::vector<uint> &ind_cols,
                    const mat2 &sub_M) {

    for (uint j = 0; j < ind_cols.size(); j++) {
        for (uint i = 0; i < ind_rows.size(); i++) {
            M(ind_rows[i], ind_cols[j]) += sub_M(i, j);
        }
    }
}

template<typename mat1, typename mat2>
void add_sub_matrix(mat1 &M,
                    const std::vector<uint> &indices,
                    const mat2 &sub_M) {

    return add_sub_matrix(M, indices, indices, sub_M);
}
