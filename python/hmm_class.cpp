#include "hmm_class.hpp"
#include "distribution_class.hpp"

#include <Python.h>
#include "structmember.h"

#include "numpy/npy_no_deprecated_api.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include <memory>

#include "python_utils.hpp"
#include "hmm.hpp"


PyTypeObject PyType_HMM = PyType_HMM_class();

static PyMemberDef PyObject_HMM_members[] = {
    // {"L", T_INT, offsetof(PyObject_HMM, hmm->L), 0,
    //  "Number of gaussians"},
    {NULL}  /* Sentinel */
};

static PyObject* hmm_get_states(PyObject_HMM *self, void *closure);
static PyObject* hmm_get_transitions(PyObject_HMM *self, void *closure);

static PyGetSetDef PyObject_HMM_getsetters[] = {
    {"states", (getter) hmm_get_states, NULL,
     "get the states of the HMM", NULL},
    {"transitions", (getter) hmm_get_transitions, NULL,
     "get transition matrix", NULL},
    {NULL}  /* Sentinel */
};

static PyObject* hmm_get_states(PyObject_HMM *self, void *closure) {

    if (self == NULL) {
        return NULL;
    }

    HMM<ftype> *hmm = self->hmm;
    if (hmm == NULL) {
        return NULL;
    }


    return NULL;
}

static PyObject* hmm_get_transitions(PyObject_HMM *self, void *closure) {

    if (self == NULL) {
        return NULL;
    }

    HMM<ftype> *hmm = self->hmm;
    if (hmm == NULL) {
        return NULL;
    }

    MatrixXR<ftype> &transition = hmm->transition;

    npy_intp dims[] = {transition.rows(), transition.cols()};

    return PyArray_SimpleNewFromData(2, dims, NPY_FTYPE, transition.data());

}

static PyObject *hmm_add_state(PyObject_HMM *self, PyObject *args, PyObject* kwargs);
static PyObject *hmm_fit(PyObject_HMM *self, PyObject *args, PyObject* kwargs);
static PyObject *hmm_viterbi(PyObject_HMM *self, PyObject *args, PyObject* kwargs);

static PyMethodDef PyObject_HMM_methods[] = {
    {"add_state", (PyCFunction) hmm_add_state, METH_VARARGS | METH_KEYWORDS, "Add an emission state"},
    {"fit", (PyCFunction) hmm_fit, METH_VARARGS | METH_KEYWORDS, "Fit method"},
    {"viterbi", (PyCFunction) hmm_viterbi, METH_VARARGS | METH_KEYWORDS, "Viterbi method"},
    {NULL}  /* Sentinel */
};


static void PyObject_HMM_dealloc(PyObject_HMM *self) {

    std::cout << "HMM_free" << std::endl;
    if (self->hmm != NULL) {
        delete self->hmm;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyObject_HMM_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        NULL};
    std::cout << "HMM_new" << std::endl;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist)) {
        return NULL;
    }

    PyObject_HMM *self = (PyObject_HMM*) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    self->hmm = new HMM<ftype>();

    return (PyObject *) self;
}


PyTypeObject PyType_HMM_class() {

    PyTypeObject result = {
        PyVarObject_HEAD_INIT(NULL, 0)
    };

    result.tp_name = "rhmm.HMM";
    result.tp_doc = "Hidden markov model";
    result.tp_basicsize = sizeof(PyObject_HMM);
    result.tp_itemsize = 0;
    result.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    result.tp_new = (newfunc) PyObject_HMM_new;

    // result.tp_init = (initproc) PyObject_HMM_init;
    result.tp_dealloc = (destructor) PyObject_HMM_dealloc;
    result.tp_members = PyObject_HMM_members;
    result.tp_methods = PyObject_HMM_methods;
    result.tp_getset = PyObject_HMM_getsetters;

    return result;
}


static PyObject *hmm_add_state(PyObject_HMM *self, PyObject *args, PyObject* kwargs) {

    if (self == NULL) {
        return NULL;
    }

    PyObject* state_obj = NULL;

    char *keywords[] = {
        "state",
        NULL
    };

    HMM<ftype> *hmm = self->hmm;
    if (hmm == NULL) {
        PyErr_SetString(PyExc_ValueError, "Internal error, you can start to panic");
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &state_obj)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }

    if (state_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "State being added is invalid");
        Py_RETURN_NONE;
    }

    if (!PyObject_TypeCheck(state_obj, &PyType_Distribution)) {
        PyErr_SetString(PyExc_ValueError, "Argument is not a valid distribution");
        Py_RETURN_NONE;
    }

    Distribution<ftype> *distribution = ((PyObject_Distribution*) state_obj)->distribution;

    if (distribution == NULL) {
        PyErr_SetString(PyExc_ValueError, "Argument distribution has not been properly initialized");
        Py_RETURN_NONE;
    }

    hmm->add_state(distribution);

    Py_RETURN_NONE;

}

static PyObject *hmm_fit(PyObject_HMM *self, PyObject *args, PyObject* kwargs) {

    if (self == NULL) {
        return NULL;
    }

    PyObject* data_obj = NULL;
    PyObject* labels_obj = NULL;
    uint max_iters = 1000;

    char *keywords[] = {
        "data",
        "labels",
        "max_iters",
        // "d",
        NULL
    };

    HMM<ftype> *hmm = self->hmm;
    if (hmm == NULL) {
        PyErr_SetString(PyExc_ValueError, "Internal error, you can start to panic");
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OI", keywords, &data_obj, &labels_obj, &max_iters)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }

    // PyObject *data_iterator = std::shared_ptr(PyObject_GetIter(data_obj), Py_DECREF_wrapper);
    PyObject *data_iterator = PyObject_GetIter(data_obj);

    if (data_iterator == NULL) {
        PyErr_SetString(PyExc_ValueError, "Data object is not iterable");
        Py_RETURN_NONE;
    }

    SCOPE_DECREF(data_iterator);
    // std::unique_ptr<PyObject, Py_DECREF_wrapper> data_iterator_ptr; // Only there for automatic DECREF

    Py_ssize_t data_len = PyObject_Length(data_obj);

    PyObject *labels_iterator = NULL;

    if (labels_obj != NULL) {

        labels_iterator = PyObject_GetIter(labels_obj);

        if (labels_iterator == NULL) {
            PyErr_SetString(PyExc_ValueError, "Label object is not iterable");
            Py_RETURN_NONE;
        }

        Py_ssize_t labels_len = PyObject_Length(labels_obj);

        if (labels_len != data_len) {
            PyErr_SetString(PyExc_ValueError, "Data and labels must have the same length");
            Py_RETURN_NONE;
        }

    }

    uint item_count = 0;
    uint lines_count = 0;

    uint ref_D = 0;

    std::vector<ftype> all_data;
    std::vector<int> all_labels;

    std::vector<uint> rows;

    for (;;) {

        uint all_data_len = all_data.size();
        uint all_labels_len = all_labels.size();

        PyObject *data_item = PyIter_Next(data_iterator);
        if (data_item == NULL) break;

        SCOPE_DECREF(data_item);

        PyArrayObject* data_array = (PyArrayObject*) PyArray_FROMANY(data_item, NPY_FTYPE, 2, 2,
                                                                     NPY_ARRAY_FORCECAST |
                                                                     NPY_ARRAY_C_CONTIGUOUS |
                                                                     NPY_ARRAY_ALIGNED);

        if (data_array == NULL) {
            PyErr_Format(PyExc_TypeError, "Sequence %d is not a valid array",
                         item_count);
            Py_RETURN_NONE;
        }

        SCOPE_DECREF(data_array);


        uint N = PyArray_DIM(data_array, 0);
        uint D = PyArray_DIM(data_array, 1);

        if (item_count == 0) {
            ref_D = D;
        }

        if (D != ref_D) {
            PyErr_Format(PyExc_TypeError,"Sequence 0 and %d don't have the same number of dimensions",
                         item_count);
            Py_RETURN_NONE;
        }

        ftype *data_array_ptr = (ftype*) PyArray_DATA(data_array);
        if (data_array_ptr == NULL) {
            PyErr_Format(PyExc_TypeError, "Sequence %d is not a valid array",
                         item_count);
            Py_RETURN_NONE;
        }

        all_data.resize(all_data_len + N * D);
        std::copy_n(data_array_ptr, N * D, all_data.begin() + all_data_len);

        if (labels_obj != NULL) {

            PyObject *labels_item = PyIter_Next(labels_iterator);
            if (labels_item == NULL) {
                PyErr_SetString(PyExc_TypeError, "Data and labels must have the same length");
                Py_RETURN_NONE;
            }

            SCOPE_DECREF(labels_item);

            PyArrayObject* labels_array = (PyArrayObject*) PyArray_FROMANY(labels_item, NPY_INT, 1, 1,
                                                                    NPY_ARRAY_FORCECAST |
                                                                    NPY_ARRAY_C_CONTIGUOUS |
                                                                    NPY_ARRAY_ALIGNED);

            SCOPE_DECREF(labels_array);

            if (labels_array == NULL) {
                PyErr_Format(PyExc_TypeError, "Sequence label %d is not a valid int array",
                             item_count);
                Py_RETURN_NONE;
            }

            if (PyArray_DIM(labels_array, 0) != N) {
                PyErr_Format(PyExc_TypeError,"Data and label length for sequence %d do not match",
                             item_count);
                Py_RETURN_NONE;
            }

            int *labels_array_ptr = (int*) PyArray_DATA(labels_array);
            if (labels_array_ptr == NULL) {
                PyErr_Format(PyExc_TypeError, "Internal numpy array error",
                             item_count);
                Py_RETURN_NONE;
            }

            all_labels.resize(all_labels_len + N);
            std::copy_n(labels_array_ptr, N, all_labels.begin() + all_labels_len);

        } else {

            all_labels.resize(all_labels_len + N);
            std::fill_n(all_labels.begin() + all_labels_len, N, -1);

        }

        rows.push_back(N);

        item_count++;
        lines_count += N;

    }

    std::vector< Sequence<ftype> > sequences;

    ftype *all_data_ptr = all_data.data();
    int *all_labels_ptr = all_labels.data();

    for (int N : rows) {
        Sequence<ftype> seq(all_data_ptr, all_labels_ptr, N, ref_D);
        sequences.push_back(seq);

        all_data_ptr   += N * ref_D;
        all_labels_ptr += N;

    }

    hmm->fit(sequences.data(), sequences.size(), max_iters);

    Py_RETURN_NONE;

}


static PyObject *hmm_viterbi(PyObject_HMM *self, PyObject *args, PyObject* kwargs) {

    if (self == NULL) {
        return NULL;
    }

    PyObject* data_obj = NULL;
    PyObject* labels_obj = NULL;

    char *keywords[] = {
        "data",
        "labels",
        // "d",
        NULL
    };

    HMM<ftype> *hmm = self->hmm;
    if (hmm == NULL) {
        PyErr_SetString(PyExc_ValueError, "Internal error, you can start to panic");
        Py_RETURN_NONE;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", keywords, &data_obj, &labels_obj)) {
        PyErr_SetString(PyExc_ValueError, "Argument parsing failed");
        Py_RETURN_NONE;
    }



    PyObject *data_iterator = PyObject_GetIter(data_obj);

    if (data_iterator == NULL) {
        PyErr_SetString(PyExc_ValueError, "Data object is not iterable");
        Py_RETURN_NONE;
    }

    SCOPE_DECREF(data_iterator);

    Py_ssize_t data_len = PyObject_Length(data_obj);

    PyObject *labels_iterator = NULL;

    if (labels_obj != NULL) {

        labels_iterator = PyObject_GetIter(labels_obj);

        if (labels_iterator == NULL) {
            PyErr_SetString(PyExc_ValueError, "Label object is not iterable");
            Py_RETURN_NONE;
        }

        Py_ssize_t labels_len = PyObject_Length(labels_obj);

        if (labels_len != data_len) {
            PyErr_SetString(PyExc_ValueError, "Data and labels must have the same length");
            Py_RETURN_NONE;
        }

    }

    uint item_count = 0;
    uint lines_count = 0;

    uint ref_D = 0;

    std::vector<ftype> all_data;
    std::vector<int> all_labels;

    std::vector<uint> rows;

    for (;;) {

        uint all_data_len = all_data.size();
        uint all_labels_len = all_labels.size();

        PyObject *data_item = PyIter_Next(data_iterator);
        if (data_item == NULL) break;

        SCOPE_DECREF(data_item);

        PyArrayObject* data_array = (PyArrayObject*) PyArray_FROMANY(data_item, NPY_FTYPE, 2, 2,
                                                                     NPY_ARRAY_FORCECAST |
                                                                     NPY_ARRAY_C_CONTIGUOUS |
                                                                     NPY_ARRAY_ALIGNED);

        if (data_array == NULL) {
            PyErr_Format(PyExc_TypeError, "Sequence %d is not a valid array",
                         item_count);
            Py_RETURN_NONE;
        }

        SCOPE_DECREF(data_array);


        uint N = PyArray_DIM(data_array, 0);
        uint D = PyArray_DIM(data_array, 1);

        if (item_count == 0) {
            ref_D = D;
        }

        if (D != ref_D) {
            PyErr_Format(PyExc_TypeError,"Sequence 0 and %d don't have the same number of dimensions",
                         item_count);
            Py_RETURN_NONE;
        }

        ftype *data_array_ptr = (ftype*) PyArray_DATA(data_array);
        if (data_array_ptr == NULL) {
            PyErr_Format(PyExc_TypeError, "Sequence %d is not a valid array",
                         item_count);
            Py_RETURN_NONE;
        }

        all_data.resize(all_data_len + N * D);
        std::copy_n(data_array_ptr, N * D, all_data.begin() + all_data_len);

        if (labels_obj != NULL) {

            PyObject *labels_item = PyIter_Next(labels_iterator);
            if (labels_item == NULL) {
                PyErr_SetString(PyExc_TypeError, "Data and labels must have the same length");
                Py_RETURN_NONE;
            }

            SCOPE_DECREF(labels_item);

            PyArrayObject* labels_array = (PyArrayObject*) PyArray_FROMANY(labels_item, NPY_INT, 1, 1,
                                                                    NPY_ARRAY_FORCECAST |
                                                                    NPY_ARRAY_C_CONTIGUOUS |
                                                                    NPY_ARRAY_ALIGNED);

            SCOPE_DECREF(labels_array);
            // std::unique_ptr<PyArrayObject, Py_DECREF_wrapper> array_ptr;

            if (labels_array == NULL) {
                PyErr_Format(PyExc_TypeError, "Sequence label %d is not a valid int array",
                             item_count);
                Py_RETURN_NONE;
            }

            if (PyArray_DIM(labels_array, 0) != N) {
                PyErr_Format(PyExc_TypeError,"Data and label length for sequence %d do not match",
                             item_count);
                Py_RETURN_NONE;
            }

            int *labels_array_ptr = (int*) PyArray_DATA(labels_array);
            if (labels_array_ptr == NULL) {
                PyErr_Format(PyExc_TypeError, "Internal numpy array error",
                             item_count);
                Py_RETURN_NONE;
            }

            all_labels.resize(all_labels_len + N);
            std::copy_n(labels_array_ptr, N, all_labels.begin() + all_labels_len);

        } else {

            all_labels.resize(all_labels_len + N);
            std::fill_n(all_labels.begin() + all_labels_len, N, -1);

        }

        rows.push_back(N);

        item_count++;
        lines_count += N;

    }

    std::vector< Sequence<ftype> > sequences;

    ftype *all_data_ptr = all_data.data();
    int *all_labels_ptr = all_labels.data();

    for (int N : rows) {
        Sequence<ftype> seq(all_data_ptr, all_labels_ptr, N, ref_D);
        sequences.push_back(seq);

        all_data_ptr   += N * ref_D;
        all_labels_ptr += N;

    }

    vector< vector<uint> > states = hmm->viterbi(sequences.data(), sequences.size());
    if (states.empty()) {
        PyErr_SetString(PyExc_ValueError, "Failed to run viterbi algorithm");
        Py_RETURN_NONE;
    }

    PyObject *result = PyList_New(sequences.size());
    if (result == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to create result list");
        Py_RETURN_NONE;
    }

    for (uint i = 0; i < states.size(); i++) {

        vector<uint> &state = states[i];

        npy_intp data_dims[] = {state.size()};
        PyObject *seq_states = PyArray_SimpleNewFromData(1, data_dims, NPY_UINT, state.transfer());

        int ok = PyList_SetItem(result, i, seq_states);
        if (ok < 0) {
            PyErr_SetString(PyExc_ValueError, "Failed to insert an element in the result list");
            Py_RETURN_NONE;
        }

    }

    return result;

}
