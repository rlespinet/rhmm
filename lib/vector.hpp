#pragma once

template<typename T>
struct vector {
    uint m_capacity;
    uint m_size;
    T *m_data;

    vector(uint capacity, uint size)
        : m_capacity(capacity)
        , m_size(size) {
        m_data = new T[capacity];
    }

    vector(uint size = 0)
        : vector(size, size) {
    }

    vector(const vector<T> &oth)
        : vector<T>(oth.m_capacity, oth.m_size) {
        std::copy(oth.m_data, oth.m_data + oth.m_size, m_data);
    }

    vector(vector<T> &&oth) {
        swap(*this, oth);
    }

    friend void swap(vector<T> &v1, vector<T> &v2) {
        using std::swap;
        swap(v1.m_capacity, v2.m_capacity);
        swap(v1.m_size, v2.m_size);
        swap(v1.m_data, v2.m_data);
    }

    vector<T>& operator=(vector<T> oth) {
        swap(*this, oth);
        return *this;
    }

    ~vector() {
        delete[] m_data;
    }

    const uint capacity() const {
        return m_capacity;
    }

    const uint size() const {
        return m_size;
    }

    const bool empty() const {
        return m_size == 0;
    }

    const T* data() const {
        return m_data;
    }

    T* data() {
        return m_data;
    }

    const T& operator[](uint i) const {
        assert(i < m_size);
        return m_data[i];
    }

    T& operator[](uint i)  {
        assert(i < m_size);
        return m_data[i];
    }

    const T& operator()(uint i) const {
        assert(i < m_size);
        return m_data[i];
    }

    T& operator()(uint i)  {
        assert(i < m_size);
        return m_data[i];
    }

    void resize(uint new_size) {
        assert(new_size <= capacity);
        this->m_size = new_size;
    }

    void inc_size(uint inc = 1) {
        assert(m_size + inc <= m_capacity);
        m_size += inc;
    }

    void dec_size(uint dec = 1) {
        assert(m_size - dec >= 0);
        m_size -= dec;
    }

    void push_back(const T& oth) {
        assert(m_size < m_capacity);
        m_data[m_size++] = oth;
    }

    void push_back(T&& oth) {
        assert(m_size < m_capacity);
        m_data[m_size++] = std::move(oth);
    }

    T&& pop_back() {
        assert(m_size > 0);
        return std::move(m_data[--m_size]);
    }

    void clear() {
        m_size = 0;
    }

    const T& last() const {
        assert(m_size > 0);
        return m_data[m_size - 1];
    }

    T& last() {
        assert(m_size > 0);
        return m_data[m_size - 1];
    }

    const T& first() const {
        assert(m_size > 0);
        return m_data[0];
    }

    T& first() {
        assert(m_size > 0);
        return m_data[0];
    }

    const T* cbegin() const {
        return m_data;
    }

    const T* cend() const {
        return m_data + m_size;
    }

    T* begin() {
        return m_data;
    }

    T* end() {
        return m_data + m_size;
    }

    // TODO(RL) Good idea ?
    T* transfer() {
        T* result = m_data;
        m_capacity = 0;
        m_size = 0;
        m_data = new T[0];
        return result;
    }

};
