#include "distribution_class.hpp"
#include "python_utils.hpp"

#include <Python.h>
#include "structmember.h"

#include "numpy/npy_no_deprecated_api.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include <memory>


////////////////////////////////////////////////////////////////////////////////
// Distribution
////////////////////////////////////////////////////////////////////////////////


PyTypeObject PyType_Distribution = PyType_Distribution_class();

static PyMemberDef PyObject_Distribution_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef PyObject_Distribution_getsetters[] = {
    {NULL}  /* Sentinel */
};

// static PyObject *distribution_multinomial(PyObject_Distribution *self,
//                                           PyObject *args, PyObject* kwargs);
// static PyObject *distribution_multivariate_gaussian(PyObject_Distribution *self,
//                                                     PyObject *args, PyObject* kwargs);

static PyMethodDef PyObject_Distribution_methods[] = {
    // {"Multinomial", (PyCFunction) distribution_multinomial,
    //  METH_STATIC | METH_VARARGS | METH_KEYWORDS, "Constructs a multinomial distribution"},
    // {"MultivariateGaussian", (PyCFunction) distribution_multivariate_gaussian,
    //  METH_STATIC | METH_VARARGS | METH_KEYWORDS, "Constructs a multivariate gaussian distribution"},
    {NULL}  /* Sentinel */
};

static void PyObject_Distribution_dealloc(PyObject_Distribution *self) {

    if (self->distribution != NULL) {
        delete self->distribution;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *PyObject_Distribution_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        NULL};

    std::cout << "new called !\n";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) {
        return NULL;
    }

    PyObject_Distribution *self = (PyObject_Distribution*) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    self->distribution = NULL;

    return (PyObject *) self;
}


PyTypeObject PyType_Distribution_class() {

    PyTypeObject result = {
        PyVarObject_HEAD_INIT(NULL, 0)
    };

    result.tp_name = "rhmm.Distribution";
    result.tp_doc = "Distribution";
    result.tp_basicsize = sizeof(PyObject_Distribution);
    result.tp_itemsize = 0;
    result.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    result.tp_new = (newfunc) PyObject_Distribution_new;

    result.tp_dealloc = (destructor) PyObject_Distribution_dealloc;
    result.tp_members = PyObject_Distribution_members;
    result.tp_methods = PyObject_Distribution_methods;
    result.tp_getset = PyObject_Distribution_getsetters;

    return result;
}


////////////////////////////////////////////////////////////////////////////////
// Multinomial
////////////////////////////////////////////////////////////////////////////////


PyTypeObject PyType_Multinomial = PyType_Multinomial_class();

static PyMemberDef PyObject_Multinomial_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject* multinomial_get_probs(PyObject_Multinomial *self, void *closure);

static PyGetSetDef PyObject_Multinomial_getsetters[] = {
    {"probs", (getter) multinomial_get_probs, NULL, "get the probability vector", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef PyObject_Multinomial_methods[] = {
    {NULL}  /* Sentinel */
};


static void PyObject_Multinomial_dealloc(PyObject_Multinomial *self);
static PyObject *PyObject_Multinomial_new(PyTypeObject *type, PyObject *args, PyObject *kwds);


PyTypeObject PyType_Multinomial_class() {

    PyTypeObject result = {
        PyVarObject_HEAD_INIT(NULL, 0)
    };

    result.tp_name = "rhmm.Multinomial";
    result.tp_doc = "Multinomial";
    result.tp_basicsize = sizeof(PyObject_Multinomial);
    result.tp_itemsize = 0;
    result.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    result.tp_new = (newfunc) PyObject_Multinomial_new;

    result.tp_dealloc = (destructor) PyObject_Multinomial_dealloc;
    result.tp_members = PyObject_Multinomial_members;
    result.tp_methods = PyObject_Multinomial_methods;
    result.tp_getset = PyObject_Multinomial_getsetters;
    result.tp_base = &PyType_Distribution;

    return result;
}

static void PyObject_Multinomial_dealloc(PyObject_Multinomial *self) {

    if (self->distribution != NULL) {
        delete self->distribution;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyObject_Multinomial_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    PyObject* obj_probs = NULL;

    char *keywords[] = {
        "probs",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", keywords, &obj_probs)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }

    PyArrayObject* probs_array = (PyArrayObject*) PyArray_FROMANY(obj_probs, NPY_FLOAT, 1, 1,
                                                                  NPY_ARRAY_FORCECAST |
                                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                                  NPY_ARRAY_ALIGNED);

    if (probs_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "probs argument is not a valid array");
        Py_RETURN_NONE;
    }

    SCOPE_DECREF(probs_array);

    uint K = PyArray_DIM(probs_array, 0);

    float *probs_array_ptr = (float*) PyArray_DATA(probs_array);
    if (probs_array_ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to recover array data");
        Py_RETURN_NONE;
    }

    VectorX<float> probs = Map< VectorX<float> >(probs_array_ptr, K);

    Distribution<float> *distribution = new Multinomial<float>(probs);
    if (distribution == NULL) {
        PyErr_SetString(PyExc_TypeError, "Allocation failed");
        Py_RETURN_NONE;
    }

    PyObject_Multinomial *self = (PyObject_Multinomial*) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    self->distribution = distribution;

    return (PyObject*) self;

}

static PyObject* multinomial_get_probs(PyObject_Multinomial *self, void *closure) {

    if (self == NULL) {
        return NULL;
    }

    Multinomial<float> *distribution = (Multinomial<float> *) self->distribution;
    if (distribution == NULL) {
        return NULL;
    }

    VectorX<float> &probs = distribution->log_probs;

    npy_intp dims[] = {probs.size()};

    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, probs.data());

}


////////////////////////////////////////////////////////////////////////////////
// MultivariateGaussian
////////////////////////////////////////////////////////////////////////////////


PyTypeObject PyType_MultivariateGaussian = PyType_MultivariateGaussian_class();

static PyMemberDef PyObject_MultivariateGaussian_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject* multivariate_gaussian_get_mean(PyObject_MultivariateGaussian *self, void *closure);
static PyObject* multivariate_gaussian_get_cov(PyObject_MultivariateGaussian *self, void *closure);

static PyGetSetDef PyObject_MultivariateGaussian_getsetters[] = {
    {"mean", (getter) multivariate_gaussian_get_mean, NULL, "get the mean parameter", NULL},
    {"cov", (getter) multivariate_gaussian_get_cov, NULL, "get the covariance parameter", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef PyObject_MultivariateGaussian_methods[] = {
    {NULL}  /* Sentinel */
};


static void PyObject_MultivariateGaussian_dealloc(PyObject_MultivariateGaussian *self);
static PyObject *PyObject_MultivariateGaussian_new(PyTypeObject *type, PyObject *args, PyObject *kwds);


PyTypeObject PyType_MultivariateGaussian_class() {

    PyTypeObject result = {
        PyVarObject_HEAD_INIT(NULL, 0)
    };

    result.tp_name = "rhmm.MultivariateGaussian";
    result.tp_doc = "MultivariateGaussian";
    result.tp_basicsize = sizeof(PyObject_MultivariateGaussian);
    result.tp_itemsize = 0;
    result.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    result.tp_new = (newfunc) PyObject_MultivariateGaussian_new;

    result.tp_dealloc = (destructor) PyObject_MultivariateGaussian_dealloc;
    result.tp_members = PyObject_MultivariateGaussian_members;
    result.tp_methods = PyObject_MultivariateGaussian_methods;
    result.tp_getset = PyObject_MultivariateGaussian_getsetters;
    result.tp_base = &PyType_Distribution;

    return result;
}

static void PyObject_MultivariateGaussian_dealloc(PyObject_MultivariateGaussian *self) {

    if (self->distribution != NULL) {
        delete self->distribution;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyObject_MultivariateGaussian_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    PyObject* obj_mean = NULL;
    PyObject* obj_cov = NULL;

    char *keywords[] = {
        "mean",
        "cov",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", keywords, &obj_mean, &obj_cov)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }

    PyArrayObject* mean_array = (PyArrayObject*) PyArray_FROMANY(obj_mean, NPY_FLOAT, 1, 1,
                                                                 NPY_ARRAY_FORCECAST |
                                                                 NPY_ARRAY_C_CONTIGUOUS |
                                                                 NPY_ARRAY_ALIGNED);

    if (mean_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "mean argument is not a valid 1D array");
        Py_RETURN_NONE;
    }

    uint D = PyArray_DIM(mean_array, 0);


    SCOPE_DECREF(mean_array);

    PyArrayObject* cov_array = (PyArrayObject*) PyArray_FROMANY(obj_cov, NPY_FLOAT, 2, 2,
                                                                NPY_ARRAY_FORCECAST |
                                                                NPY_ARRAY_C_CONTIGUOUS |
                                                                NPY_ARRAY_ALIGNED);

    if (cov_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "covariance argument is not a valid 2D array");
        Py_RETURN_NONE;
    }

    SCOPE_DECREF(cov_array);

    if (PyArray_DIM(cov_array, 0) != D || PyArray_DIM(cov_array, 1) != D) {
        PyErr_SetString(PyExc_ValueError, "dimensions of mean and covariance array do not match");
        Py_RETURN_NONE;
    }

    float *mean_array_ptr = (float*) PyArray_DATA(mean_array);
    if (mean_array_ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to recover mean array data");
        Py_RETURN_NONE;
    }

    VectorX<float> mean = Map< VectorX<float> >(mean_array_ptr, D);

    float *cov_array_ptr = (float*) PyArray_DATA(cov_array);
    if (cov_array_ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to recover covariance array data");
        Py_RETURN_NONE;
    }

    MatrixX<float> cov = Map< MatrixX<float> >(cov_array_ptr, D, D);


    Distribution<float> *distribution = new MultivariateGaussian<float>(mean, cov);
    if (distribution == NULL) {
        PyErr_SetString(PyExc_TypeError, "Allocation failed");
        Py_RETURN_NONE;
    }

    PyObject_MultivariateGaussian *self = (PyObject_MultivariateGaussian*) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    self->distribution = distribution;

    return (PyObject*) self;

}

static PyObject* multivariate_gaussian_get_mean(PyObject_MultivariateGaussian *self, void *closure) {

    if (self == NULL) {
        return NULL;
    }

    MultivariateGaussian<float> *distribution = (MultivariateGaussian<float> *) self->distribution;
    if (distribution == NULL) {
        return NULL;
    }

    VectorX<float> &mean = distribution->mean;

    npy_intp dims[] = {mean.size()};

    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, mean.data());

}

static PyObject* multivariate_gaussian_get_cov(PyObject_MultivariateGaussian *self, void *closure) {

    if (self == NULL) {
        return NULL;
    }

    MultivariateGaussian<float> *distribution = (MultivariateGaussian<float> *) self->distribution;
    if (distribution == NULL) {
        return NULL;
    }

    MatrixX<float> &cov = distribution->cov;

    npy_intp dims[] = {cov.rows(), cov.cols()};

    return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, cov.data());

}
