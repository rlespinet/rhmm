#pragma once

#include <Python.h>

#include "distribution.hpp"


struct PyObject_Distribution {
    PyObject_HEAD;
    Distribution<float> *distribution;
};

PyTypeObject PyType_Distribution_class();

extern PyTypeObject PyType_Distribution;
