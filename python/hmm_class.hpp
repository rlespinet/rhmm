#pragma once

#include <Python.h>

#include "ftype.hpp"
#include "hmm.hpp"

struct PyObject_HMM {
    PyObject_HEAD;
    HMM<ftype> *hmm;
};

PyTypeObject PyType_HMM_class();

extern PyTypeObject PyType_HMM;
