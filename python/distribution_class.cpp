#include "distribution_class.hpp"
#include "python_utils.hpp"

#include <Python.h>
#include "structmember.h"

#include "numpy/npy_no_deprecated_api.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include <memory>

PyTypeObject PyType_Distribution = PyType_Distribution_class();

static PyMemberDef PyObject_Distribution_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef PyObject_Distribution_getsetters[] = {
    {NULL}  /* Sentinel */
};

static PyObject *distribution_multinomial(PyObject_Distribution *self,
                                          PyObject *args, PyObject* kwargs);
static PyObject *distribution_multivariate_gaussian(PyObject_Distribution *self,
                                                    PyObject *args, PyObject* kwargs);

static PyMethodDef PyObject_Distribution_methods[] = {
    {"Multinomial", (PyCFunction) distribution_multinomial,
     METH_STATIC | METH_VARARGS | METH_KEYWORDS, "Constructs a multinomial distribution"},
    {"MultivariateGaussian", (PyCFunction) distribution_multivariate_gaussian,
     METH_STATIC | METH_VARARGS | METH_KEYWORDS, "Constructs a multivariate gaussian distribution"},
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


static PyObject *distribution_multinomial(PyObject_Distribution *self,
                                          PyObject *args, PyObject* kwargs) {

    if (self != NULL) {
        PyErr_SetString(PyExc_ValueError, "Internal error, static method called on object");
        Py_RETURN_NONE;
    }

    PyObject* obj_probs = NULL;

    char *keywords[] = {
        "probs",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &obj_probs)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }

    PyArrayObject* probs_array = (PyArrayObject*) PyArray_FROMANY(obj_probs, NPY_FLOAT, 1, 1,
                                                                  NPY_ARRAY_FORCECAST |
                                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                                  NPY_ARRAY_ALIGNED);

    if (probs_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Data object is not iterable");
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

    // Construct object from data
    PyObject *empty_tuple = PyTuple_New(0);
    if (distribution == NULL) {
        PyErr_SetString(PyExc_TypeError, "Internal error : Failed to create empty tuple");
        Py_RETURN_NONE;
    }

    SCOPE_DECREF(empty_tuple);

    PyObject *obj = PyObject_CallObject((PyObject *) &PyType_Distribution, empty_tuple);
    if (obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "Internal error : Failed to create associated python object");
        Py_RETURN_NONE;
    }

    ((PyObject_Distribution*)obj)->distribution = distribution;

    return obj;

}
