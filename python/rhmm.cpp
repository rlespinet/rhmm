#include <Python.h>
#include "structmember.h"

#include <iostream>
#include <cstring>

#include "numpy/npy_no_deprecated_api.h"

#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include "rhmm.hpp"

#include "hmm_class.hpp"
#include "distribution_class.hpp"

static char module_docstring[] = "Functions related to machine learning";

// static char ppca_docstring [] = "Probabilistic PCA";
static char hmm_state_docstring [] = "Type representing an emission state of the HMM";
static char hmm_docstring [] = "HMM implementation";

// static PyObject *rhmm_ppca_bind(PyObject *self, PyObject *args, PyObject *kwargs);
// static PyObject *rhmm_hmm_bind(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef rhmm_methods[] = {
    // {"ppca", (PyCFunction) rhmm_ppca_bind, METH_VARARGS | METH_KEYWORDS, ppca_docstring},
    // {"hmm", (PyCFunction) rhmm_hmm_bind, METH_VARARGS | METH_KEYWORDS, hmm_docstring},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef rhmm_module = {
    PyModuleDef_HEAD_INIT,
    "rhmm",
    module_docstring,
    -1,
    rhmm_methods
};


// PyTypeObject PyType_HMM_State = PyType_State_class();
// PyTypeObject PyType_Distribution_State = PyType_Distribution_class();

PyMODINIT_FUNC PyInit_rhmm(void) {
    import_array();

    PyObject *m = PyModule_Create(&rhmm_module);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&PyType_HMM) < 0)
        return NULL;

    if (PyType_Ready(&PyType_Distribution) < 0)
        return NULL;

    Py_INCREF(&PyType_HMM);
    Py_INCREF(&PyType_Distribution);

    PyModule_AddObject(m, "HMM", (PyObject *) &PyType_HMM);
    PyModule_AddObject(m, "Distribution", (PyObject *) &PyType_Distribution);

    return m;
}
