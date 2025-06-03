#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "kernel_interface.h"
#include <stdio.h>
#include <stdlib.h>
//#include <numpy/arrayobject.h>
//#include <complex.h>

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
    PyObject *n_points_obj = PyObject_GetAttrString(kernel_instance, "n_points");
    PyObject *rhs_obj = PyObject_GetAttrString(kernel_instance, "rhs");

    if (!points_obj) {
        printf("points_obj is NULL\n");
        return fail(k);
    }
    if (!PyArray_Check(points_obj)) {
        printf("points_obj is not a NumPy array\n");
        return fail(k);
    }
    if (!mat_obj) {
        printf("mat_obj is NULL\n");
        return fail(k);
    }
    if (!n_points_obj) {
        printf("n_points_obj is NULL\n");
        return fail(k);
    }
    if (!PyLong_Check(n_points_obj)) {
        printf("n_points_obj is not a Python int\n");
        return fail(k);
    }
    if (!rhs_obj) {
        printf("rhs_obj is NULL\n");
        return fail(k);
    }
    k->points = (PyArrayObject *)points_obj;
    k->mat = (PyArrayObject *)mat_obj;
    k->n_points = (int)PyLong_AsLong(n_points_obj);
    k->rhs = (PyArrayObject *)rhs_obj;
    return k;
}

Kernel* initialize_kernel(const char* python_executable, const char *class_name, double arg1, const char *geometry_type, double kappa) {
    
    if (!Py_IsInitialized()) {
        wchar_t *program_name = Py_DecodeLocale(python_executable, NULL);
        Py_SetProgramName(program_name);

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

    PyObject *arg1_obj;
    if ((int)arg1 == arg1) {
        arg1_obj = PyLong_FromLong((long)arg1);
    } else {
        arg1_obj = PyFloat_FromDouble(arg1);
    }

    // Create Python string object for geometry_type
    PyObject *geometry_obj = PyUnicode_FromString(geometry_type);

    // Create argument tuple with (arg1, geometry_type, kappa)
    PyObject *args = PyTuple_Pack(3, arg1_obj, geometry_obj, PyFloat_FromDouble(kappa));

    Py_DECREF(arg1_obj);
    Py_DECREF(geometry_obj);


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


int mv_kernel_real(Kernel *k, const double *input, double *output, int len) {
    if (!k || k->n_points != len) return 0;

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

// input/output are arrays of double complex
int mv_kernel_complex(Kernel *k, const double _Complex *input, double _Complex *output, int len) {
    if (!k || k->n_points != len) return 0;

    npy_intp dims[1] = {len};
    PyObject *v = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    if (!v) return 0;

    // Copy input complex values into NumPy array
    memcpy(PyArray_DATA((PyArrayObject *)v), input, len * sizeof(double _Complex));

    // Call Python method
    PyObject *result = PyObject_CallMethod(k->pyobj, "mv", "O", v);
    Py_DECREF(v);
    if (!result || !PyArray_Check(result)) {
        Py_XDECREF(result);
        PyErr_Print();
        return 0;
    }

    PyArrayObject *result_arr = (PyArrayObject *)result;

    if (PyArray_NDIM(result_arr) != 1 || PyArray_DIM(result_arr, 0) != len ||
        PyArray_TYPE(result_arr) != NPY_COMPLEX128) {
        Py_DECREF(result);
        return 0;
    }

    // Copy result to output
    memcpy(output, PyArray_DATA(result_arr), len * sizeof(double _Complex));
    Py_DECREF(result);
    return 1;
}


const double* get_points(Kernel *k) {
    if (!k || !k->points) return NULL;

    if (PyArray_NDIM(k->points) != 2 || PyArray_DIM(k->points, 1) != 3)
        return NULL;

    return (const double *)PyArray_DATA(k->points);
}

double get_condition_number(Kernel *k) {
    if (!k || !k->pyobj) {
        fprintf(stderr, "Invalid kernel object\n");
        return -1.0;
    }

    PyObject *result = PyObject_CallMethod(k->pyobj, "get_cond", NULL);
    if (!result) {
        PyErr_Print();
        return -1.0;
    }

    double cond_number = PyFloat_AsDouble(result);
    Py_DECREF(result);

    if (PyErr_Occurred()) {
        PyErr_Print();
        return -1.0;
    }

    return cond_number;
}

size_t get_n_points(Kernel *k) {
    if (!k) return (size_t)-1;  // sentinel value for error
    return (size_t)(k->n_points);
}


const double _Complex* kernel_get_complex_rhs(Kernel *k) {
    if (!k || !k->rhs) return NULL;

    if (PyArray_NDIM(k->points) != 2)
        return NULL;

    return (const double _Complex *)PyArray_DATA(k->rhs);

}


const double * kernel_get_real_rhs(Kernel *k) {
    if (!k || !k->rhs) return NULL;

    if (PyArray_NDIM(k->points) != 2)
        return NULL;

    return (const double *)PyArray_DATA(k->rhs);
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
