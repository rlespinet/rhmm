#pragma once

#include <Python.h>

#include "ftype.hpp"
#include "distribution.hpp"

struct PyObject_Distribution {
    PyObject_HEAD;
    Distribution<ftype> *distribution;
};

PyTypeObject PyType_Distribution_class();

extern PyTypeObject PyType_Distribution;

struct PyObject_Multinomial : public PyObject_Distribution {
    PyObject_HEAD;
};

PyTypeObject PyType_Multinomial_class();

extern PyTypeObject PyType_Multinomial;


struct PyObject_MultivariateGaussian : public PyObject_Distribution {
    PyObject_HEAD;
};

PyTypeObject PyType_MultivariateGaussian_class();

extern PyTypeObject PyType_MultivariateGaussian;
