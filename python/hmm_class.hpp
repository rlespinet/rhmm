#pragma once

#include <Python.h>

#include "hmm.hpp"

struct PyObject_HMM {
    PyObject_HEAD;
    HMM<float> *hmm;
};

PyTypeObject PyType_HMM_class();

extern PyTypeObject PyType_HMM;
