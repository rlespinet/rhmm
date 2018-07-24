#pragma once

struct Py_DECREF_wrapper {
    void operator()(PyObject *py_obj) {
        Py_DECREF(py_obj);
    }
};

#define SCOPE_DECREF(ptr)                                               \
    std::unique_ptr<PyObject, Py_DECREF_wrapper> DECREF_##ptr((PyObject*) ptr)
