#pragma once

#include "common.hpp"

#include <initializer_list>

template<typename T, uint D>
struct ndarray {
    T *m_data;
    uint m_size;
    uint m_dims[D];

    ndarray()
        : m_data(nullptr)
        , m_size(0)
        , m_dims() {
    }

    template<typename ...Args>
    ndarray(Args... args) {
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        init_dim(0, args...);
        m_size = compute_storage(args...);
        m_data = new T[m_size];
    }

    void init_dim(uint) {}

    template<typename ...Args>
    void init_dim(uint i, uint first, Args... args) {
        m_dims[i] = first;
        init_dim(i+1, args...);
    }

    uint compute_storage() {return 1;}

    template<typename ...Args>
    uint compute_storage(uint first, Args... args) {
        return first * compute_storage(args...);
    }

    ndarray(const ndarray& oth) {
        // std::cout << "sq_ndarray_copy" << std::endl;
        assert(false);
    }

    ndarray(ndarray &&oth) {
        // std::cout << "ndarray_move" << std::endl;
        swap(*this, oth);
    }

    ~ndarray() {
        // std::cout << "ndarray_delete" << std::endl;
        if (m_data != nullptr) {
            delete[] m_data;
        }
    }

    ndarray& operator=(ndarray oth) {
        // std::cout << "sq_ndarray_eq" << std::endl;
        swap(*this, oth);
        return *this;
    }

    friend void swap(ndarray &m1, ndarray &m2) {
        using std::swap;
        // std::cout << "ndarray_swap" << std::endl;
        swap(m1.m_dims, m2.m_dims);
        swap(m1.m_data, m2.m_data);
        swap(m1.m_size, m2.m_size);
    }

    const T& operator[](uint id) const {
        return this->m_data[id];
    }

    T& operator[](uint id) {
        return this->m_data[id];
    }

    template<typename ...Args>
    const T& operator()(Args... args) const {
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        uint id = compute_id(0, args...);
        return this->m_data[id];
    }

    template<typename ...Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == D, "Wrong number of dimensions");
        uint id = compute_id(0, args...);
        return this->m_data[id];
    }

    const uint compute_id(uint) const {return 0;}

    template<typename ...Args>
    const uint compute_id(uint i, uint first, Args... args) const {
        return first + m_dims[i] * compute_id(i+1, args...);
    }

    const uint shape(uint i) const {
        static_assert(i < D, "Wrong number of dimensions");
        return m_dims[i];
    }

    const uint size() const {
        return m_size;
    }

    const T* data() const {
        return this->m_data;
    }

    T* data() {
        return this->m_data;
    }

    void fill(const T &elt) {
        for (int i = 0; i < m_size; i++) {
            m_data[i] = elt;
        }
    }

};
