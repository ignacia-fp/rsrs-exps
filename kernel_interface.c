#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "kernel_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

Kernel* fail(Kernel *k) {
    PyErr_Print();
    free(k);
    return NULL;
}

Kernel* create_kernel(PyObject *kernel_instance) {
    Kernel *k = (Kernel *)malloc(sizeof(Kernel));
    if (!k) return NULL;

    PyObject *points_obj = PyObject_GetAttrString(kernel_instance, "points");
    PyObject *mat_obj = PyObject_GetAttrString(kernel_instance, "mat");
    PyObject *dim_obj = PyObject_GetAttrString(kernel_instance, "dim");

    if ((!points_obj || !PyArray_Check(points_obj)) || 
        (!mat_obj || !PyArray_Check(mat_obj)) ||
        (!dim_obj || !PyLong_Check(dim_obj))) {
        return fail(k);
    }

    k->points = (PyArrayObject *)points_obj;
    k->mat = (PyArrayObject *)mat_obj;
    k->dim = (int)PyLong_AsLong(dim_obj);

    return k;
}

Kernel* initialize_kernel(const char *class_name, int dim, double kappa) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (_import_array() < 0) {
            PyErr_Print();
            return NULL;
        }
    }

    PyRun_SimpleString("import sys; sys.path.append('.')");

    PyObject *module = PyImport_ImportModule("python_kernels");
    if (!module) {
        PyErr_Print();
        return NULL;
    }

    PyObject *kernel_class = PyObject_GetAttrString(module, class_name);
    Py_DECREF(module);
    if (!kernel_class) {
        PyErr_Print();
        return NULL;
    }

    PyObject *args = PyTuple_Pack(2, PyLong_FromLong(dim), PyFloat_FromDouble(kappa));
    PyObject *instance = PyObject_CallObject(kernel_class, args);
    Py_DECREF(args);
    Py_DECREF(kernel_class);
    if (!instance) {
        PyErr_Print();
        return NULL;
    }

    Kernel *k = create_kernel(instance);
    if (!k) return NULL;
    k->pyobj = instance;  // take ownership
    return k;
}


int mv_kernel(Kernel *k, const double *input, double *output, int len) {
    if (!k || k->dim != len) return 0;

    npy_intp dims[1] = {len};
    PyObject *v = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!v) return 0;

    memcpy(PyArray_DATA((PyArrayObject *)v), input, len * sizeof(double));
    PyObject *result = PyObject_CallMethod(k->pyobj, "mv", "O", v);
    Py_DECREF(v);
    if (!result || !PyArray_Check(result)) {
        Py_XDECREF(result);
        PyErr_Print();
        return 0;
    }

    PyArrayObject *result_arr = (PyArrayObject *)result;
    if (PyArray_NDIM(result_arr) != 1 || PyArray_DIM(result_arr, 0) != len) {
        Py_DECREF(result);
        return 0;
    }

    memcpy(output, PyArray_DATA(result_arr), len * sizeof(double));
    Py_DECREF(result);
    return 1;
}


const double* get_points(Kernel *k) {
    if (!k || !k->points) return NULL;

    if (PyArray_NDIM(k->points) != 2 || PyArray_DIM(k->points, 1) != 3)
        return NULL;

    return (const double *)PyArray_DATA(k->points);
}


void finalize_kernel(Kernel *k) {
    if (!k) return;
    Py_DECREF(k->points);
    Py_DECREF(k->mat);
    Py_DECREF(k->pyobj);
    free(k);
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}
