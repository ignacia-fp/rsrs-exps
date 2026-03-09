#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "structured_operator_interface.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

// -------------------------
// Helper for failure
// -------------------------
static StructuredOperator* fail(StructuredOperator* k) {
  PyErr_Print();
  free(k);
  return NULL;
}

// -------------------------
// Create operator
// -------------------------
StructuredOperator* create_structured_operator(PyObject* instance) {
  StructuredOperator* k =
      (StructuredOperator*)malloc(sizeof(StructuredOperator));
  if (!k) return NULL;

  PyObject* points_obj = PyObject_GetAttrString(instance, "points");
  PyObject* mat_obj = PyObject_GetAttrString(instance, "mat");
  PyObject* n_points_obj = PyObject_GetAttrString(instance, "n_points");
  PyObject* rhs_obj = PyObject_GetAttrString(instance, "rhs");

  if (!points_obj || !PyArray_Check(points_obj)) return fail(k);
  if (!mat_obj || !n_points_obj || !rhs_obj) return fail(k);
  if (!PyLong_Check(n_points_obj)) return fail(k);

  k->points = (PyArrayObject*)points_obj;
  k->mat = (PyArrayObject*)mat_obj;
  k->n_points = (int)PyLong_AsLong(n_points_obj);
  k->pyobj = instance;
  Py_INCREF(k->pyobj);

  // Always wrap RHS as list
  if (PyList_Check(rhs_obj)) {
    k->rhs_list = rhs_obj;
    k->n_rhs = (int)PyList_Size(rhs_obj);
    Py_INCREF(rhs_obj);
  } else if (PyArray_Check(rhs_obj)) {
    PyObject* list = PyList_New(1);
    Py_INCREF(rhs_obj);
    PyList_SetItem(list, 0, rhs_obj);  // steals reference to rhs_obj
    k->rhs_list = list;
    k->n_rhs = 1;
  } else {
    return fail(k);
  }

  return k;
}

// -------------------------
// Initialize operator
// -------------------------
StructuredOperator* initialize_structured_operator(
    const char* python_executable, const char* class_name, double arg1,
    const char* geometry_type, double kappa, const char* precision,
    int n_sources, int init_samples, const char* assembler) {
  if (!Py_IsInitialized()) {
    wchar_t* program_name = Py_DecodeLocale(python_executable, NULL);
    if (!program_name) return NULL;

    Py_SetProgramName(program_name);
    Py_Initialize();
    PyMem_RawFree(program_name);

    if (_import_array() < 0) return NULL;
  }

  PyRun_SimpleString("import sys; sys.path.append('.')");

  PyObject* module = PyImport_ImportModule("python.structured_operators");
  if (!module) return NULL;

  PyObject* cls = PyObject_GetAttrString(module, class_name);
  Py_DECREF(module);
  if (!cls) return NULL;

  PyObject* arg1_obj = ((int)arg1 == arg1) ? PyLong_FromLong((long)arg1)
                                           : PyFloat_FromDouble(arg1);

  PyObject* kappa_obj = PyFloat_FromDouble(kappa);
  PyObject* geom_obj = PyUnicode_FromString(geometry_type);
  PyObject* prec_obj = PyUnicode_FromString(precision);
  PyObject* ns_obj = PyLong_FromLong(n_sources);
  PyObject* init_obj = PyLong_FromLong(init_samples);
  PyObject* assembler_obj = PyUnicode_FromString(assembler);

  PyObject* args = PyTuple_Pack(7, arg1_obj, kappa_obj, geom_obj, prec_obj,
                                ns_obj, init_obj, assembler_obj);

  Py_DECREF(arg1_obj);
  Py_DECREF(kappa_obj);
  Py_DECREF(geom_obj);
  Py_DECREF(prec_obj);
  Py_DECREF(ns_obj);
  Py_DECREF(init_obj);
  Py_DECREF(assembler_obj);

  PyObject* instance = PyObject_CallObject(cls, args);
  Py_DECREF(args);
  Py_DECREF(cls);

  if (!instance) return NULL;

  return create_structured_operator(instance);
}

// -------------------------
// Matrix-vector multiplications
// -------------------------
int mv_structured_operator_real(StructuredOperator* k, const double* input,
                                double* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)input);
  if (!input_array) return 0;

  PyObject* result = PyObject_CallMethod(k->pyobj, "mv", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    Py_DECREF(result);
    fprintf(stderr, "Error: mv() did not return a numpy array\n");
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "Error: mv() output has incorrect shape\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_DOUBLE) {
    fprintf(stderr, "Error: mv() output has wrong dtype (expected float64)\n");
    Py_DECREF(result);
    return 0;
  }

  double* data = (double*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];

  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_complex(StructuredOperator* k,
                                   const double _Complex* input,
                                   double _Complex* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_CDOUBLE, (void*)input);
  if (!input_array) return 0;

  PyObject* result = PyObject_CallMethod(k->pyobj, "mv", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    Py_DECREF(result);
    fprintf(stderr, "Error: mv() did not return a numpy array\n");
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "Error: mv() output has incorrect shape\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_CDOUBLE) {
    fprintf(stderr,
            "Error: mv() output has wrong dtype (expected complex128)\n");
    Py_DECREF(result);
    return 0;
  }

  double _Complex* data = (double _Complex*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];

  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_real32(StructuredOperator* k, const float* input,
                                  float* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};

  // Create NumPy array that views the input f32 buffer (no copy)
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)input);
  if (!input_array) return 0;

  PyObject* result = PyObject_CallMethod(k->pyobj, "mv", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    Py_DECREF(result);
    fprintf(stderr, "Error: mv() did not return a numpy array\n");
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "Error: mv() output has incorrect shape\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    fprintf(stderr, "Error: mv() output has wrong dtype (expected float32)\n");
    Py_DECREF(result);
    return 0;
  }

  float* data = (float*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) {
    output[i] = data[i];
  }

  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_complex32(StructuredOperator* k,
                                     const float _Complex* input,
                                     float _Complex* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, (void*)input);
  if (!input_array) return 0;

  PyObject* result = PyObject_CallMethod(k->pyobj, "mv", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    Py_DECREF(result);
    fprintf(stderr, "Error: mv() did not return a numpy array\n");
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "Error: mv() output has incorrect shape\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_CFLOAT) {
    fprintf(stderr,
            "Error: mv() output has wrong dtype (expected complex64)\n");
    Py_DECREF(result);
    return 0;
  }

  float _Complex* data = (float _Complex*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) {
    output[i] = data[i];
  }

  Py_DECREF(result);
  return 1;
}

// ================================
// Transposed matrix-vector product
// ================================
int mv_structured_operator_real_trans(StructuredOperator* k,
                                      const double* input, double* output,
                                      int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)input);
  if (!input_array) return 0;

  // Call Python method "mv_trans"
  PyObject* result =
      PyObject_CallMethod(k->pyobj, "mv_trans", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }
  if (!PyArray_Check(result)) {
    fprintf(stderr, "mv_trans() did not return numpy array\n");
    Py_DECREF(result);
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "mv_trans() output wrong length\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_DOUBLE) {
    fprintf(stderr, "mv_trans() output wrong dtype (expected float64)\n");
    Py_DECREF(result);
    return 0;
  }

  double* data = (double*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];

  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_complex_trans(StructuredOperator* k,
                                         const double _Complex* input,
                                         double _Complex* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_CDOUBLE, (void*)input);
  if (!input_array) return 0;

  PyObject* result =
      PyObject_CallMethod(k->pyobj, "mv_trans", "O", input_array);
  Py_DECREF(input_array);

  if (!result) {
    PyErr_Print();
    return 0;
  }
  if (!PyArray_Check(result)) {
    fprintf(stderr, "mv_trans() did not return numpy array\n");
    Py_DECREF(result);
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "mv_trans() output wrong length\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_CDOUBLE) {
    fprintf(stderr, "mv_trans() output wrong dtype (expected complex128)\n");
    Py_DECREF(result);
    return 0;
  }

  double _Complex* data = (double _Complex*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];
  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_real32_trans(StructuredOperator* k,
                                        const float* input, float* output,
                                        int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)input);
  if (!input_array) return 0;

  PyObject* result =
      PyObject_CallMethod(k->pyobj, "mv_trans", "O", input_array);
  Py_DECREF(input_array);
  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    fprintf(stderr, "mv_trans() did not return numpy array\n");
    Py_DECREF(result);
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "mv_trans() output wrong length\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    fprintf(stderr, "mv_trans() output wrong dtype (expected float32)\n");
    Py_DECREF(result);
    return 0;
  }

  float* data = (float*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];

  Py_DECREF(result);
  return 1;
}

int mv_structured_operator_complex32_trans(StructuredOperator* k,
                                           const float _Complex* input,
                                           float _Complex* output, int len) {
  if (!k || !k->pyobj) return 0;

  npy_intp dims[1] = {len};
  PyObject* input_array =
      PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, (void*)input);
  if (!input_array) return 0;

  PyObject* result =
      PyObject_CallMethod(k->pyobj, "mv_trans", "O", input_array);
  Py_DECREF(input_array);
  if (!result) {
    PyErr_Print();
    return 0;
  }

  if (!PyArray_Check(result)) {
    fprintf(stderr, "mv_trans() did not return numpy array\n");
    Py_DECREF(result);
    return 0;
  }

  PyArrayObject* arr = (PyArrayObject*)result;
  if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != len) {
    fprintf(stderr, "mv_trans() output wrong length\n");
    Py_DECREF(result);
    return 0;
  }

  if (PyArray_TYPE(arr) != NPY_CFLOAT) {
    fprintf(stderr, "mv_trans() output wrong dtype (expected complex64)\n");
    Py_DECREF(result);
    return 0;
  }

  float _Complex* data = (float _Complex*)PyArray_DATA(arr);
  for (int i = 0; i < len; ++i) output[i] = data[i];

  Py_DECREF(result);
  return 1;
}

// -------------------------
// Geometry and condition
// -------------------------
const double* get_points(StructuredOperator* k) {
  if (!k || !k->points) return NULL;
  if (PyArray_NDIM(k->points) != 2 || PyArray_DIM(k->points, 1) != 3)
    return NULL;

  if (PyArray_TYPE(k->points) != NPY_DOUBLE) {
    fprintf(stderr, "Error: points is not float64\n");
    return NULL;
  }

  if (!PyArray_ISCARRAY(k->points)) {
    fprintf(stderr, "Error: points is not contiguous/aligned\n");
    return NULL;
  }

  return (const double*)PyArray_DATA(k->points);
}

size_t get_n_points(StructuredOperator* k) {
  return k ? (size_t)k->n_points : 0;
}

double get_condition_number(StructuredOperator* k) {
  if (!k || !k->pyobj) return -1.0;
  PyObject* res = PyObject_CallMethod(k->pyobj, "cond", NULL);
  if (!res) return -1.0;
  double c = PyFloat_AsDouble(res);
  Py_DECREF(res);
  return c;
}

// -------------------------
// Retrieve all RHS at once
// -------------------------
const double** get_all_real_rhs(StructuredOperator* k, int* n_rhs,
                                int* len_out) {
  if (!k || !k->rhs_list) return NULL;
  *n_rhs = k->n_rhs;
  *len_out = k->n_points;

  const double** ptrs = (const double**)malloc(k->n_rhs * sizeof(double*));
  if (!ptrs) return NULL;

  for (int i = 0; i < k->n_rhs; ++i) {
    PyObject* arr = PyList_GetItem(k->rhs_list, i);  // borrowed ref
    if (!PyArray_Check(arr)) {
      fprintf(stderr, "Error: RHS[%d] is not a numpy array\n", i);
      free(ptrs);
      return NULL;
    }

    PyArrayObject* a = (PyArrayObject*)arr;

    if (PyArray_NDIM(a) != 1 || PyArray_DIM(a, 0) != k->n_points) {
      fprintf(stderr, "Error: RHS[%d] has wrong shape\n", i);
      free(ptrs);
      return NULL;
    }

    if (PyArray_TYPE(a) != NPY_DOUBLE) {
      fprintf(stderr, "Error: RHS[%d] is not float64\n", i);
      free(ptrs);
      return NULL;
    }

    if (!PyArray_ISCARRAY(a)) {
      fprintf(stderr, "Error: RHS[%d] is not contiguous/aligned\n", i);
      free(ptrs);
      return NULL;
    }

    ptrs[i] = (const double*)PyArray_DATA(a);
  }

  return ptrs;
}

const double _Complex** get_all_complex_rhs(StructuredOperator* k, int* n_rhs,
                                            int* len_out) {
  if (!k || !k->rhs_list) return NULL;
  *n_rhs = k->n_rhs;
  *len_out = k->n_points;

  const double _Complex** ptrs =
      (const double _Complex**)malloc(k->n_rhs * sizeof(double _Complex*));
  if (!ptrs) return NULL;

  for (int i = 0; i < k->n_rhs; ++i) {
    PyObject* arr = PyList_GetItem(k->rhs_list, i);  // borrowed ref
    if (!PyArray_Check(arr)) {
      fprintf(stderr, "Error: RHS[%d] is not a numpy array\n", i);
      free(ptrs);
      return NULL;
    }

    PyArrayObject* a = (PyArrayObject*)arr;

    if (PyArray_NDIM(a) != 1 || PyArray_DIM(a, 0) != k->n_points) {
      fprintf(stderr, "Error: RHS[%d] has wrong shape\n", i);
      free(ptrs);
      return NULL;
    }

    if (PyArray_TYPE(a) != NPY_CDOUBLE) {
      fprintf(stderr, "Error: complex RHS[%d] is not complex128\n", i);
      free(ptrs);
      return NULL;
    }

    if (!PyArray_ISCARRAY(a)) {
      fprintf(stderr, "Error: RHS[%d] is not contiguous/aligned\n", i);
      free(ptrs);
      return NULL;
    }

    ptrs[i] = (const double _Complex*)PyArray_DATA(a);
  }

  return ptrs;
}

const float** get_all_real_rhs_f32(StructuredOperator* k, int* n_rhs,
                                   int* len_out) {
  if (!k || !k->rhs_list) return NULL;

  *n_rhs = k->n_rhs;
  *len_out = k->n_points;

  const float** ptrs = (const float**)malloc(k->n_rhs * sizeof(float*));
  if (!ptrs) return NULL;

  for (int i = 0; i < k->n_rhs; ++i) {
    PyObject* arr = PyList_GetItem(k->rhs_list, i);  // borrowed ref
    if (!PyArray_Check(arr)) {
      fprintf(stderr, "Error: RHS[%d] is not a numpy array\n", i);
      free(ptrs);
      return NULL;
    }

    PyArrayObject* a = (PyArrayObject*)arr;
    if (PyArray_NDIM(a) != 1 || PyArray_DIM(a, 0) != k->n_points) {
      fprintf(stderr, "Error: RHS[%d] has wrong shape\n", i);
      free(ptrs);
      return NULL;
    }

    if (PyArray_TYPE(a) != NPY_FLOAT32) {
      fprintf(stderr, "Error: RHS[%d] is not float32\n", i);
      free(ptrs);
      return NULL;
    }

    if (!PyArray_ISCARRAY(a)) {
      fprintf(stderr, "Error: RHS[%d] is not contiguous/aligned\n", i);
      free(ptrs);
      return NULL;
    }

    ptrs[i] = (const float*)PyArray_DATA(a);
  }

  return ptrs;
}

const float _Complex** get_all_complex_rhs_f32(StructuredOperator* k,
                                               int* n_rhs, int* len_out) {
  if (!k || !k->rhs_list) return NULL;

  *n_rhs = k->n_rhs;
  *len_out = k->n_points;

  const float _Complex** ptrs =
      (const float _Complex**)malloc(k->n_rhs * sizeof(float _Complex*));
  if (!ptrs) return NULL;

  for (int i = 0; i < k->n_rhs; ++i) {
    PyObject* arr = PyList_GetItem(k->rhs_list, i);  // borrowed ref
    if (!PyArray_Check(arr)) {
      fprintf(stderr, "Error: RHS[%d] is not a numpy array\n", i);
      free(ptrs);
      return NULL;
    }

    PyArrayObject* a = (PyArrayObject*)arr;
    if (PyArray_NDIM(a) != 1 || PyArray_DIM(a, 0) != k->n_points) {
      fprintf(stderr, "Error: RHS[%d] has wrong shape\n", i);
      free(ptrs);
      return NULL;
    }

    if (PyArray_TYPE(a) != NPY_CFLOAT) {
      fprintf(stderr, "Error: complex RHS[%d] is not complex64\n", i);
      free(ptrs);
      return NULL;
    }

    if (!PyArray_ISCARRAY(a)) {
      fprintf(stderr, "Error: RHS[%d] is not contiguous/aligned\n", i);
      free(ptrs);
      return NULL;
    }

    ptrs[i] = (const float _Complex*)PyArray_DATA(a);
  }

  return ptrs;
}

// -------------------------
// Clean up
// -------------------------
void finalize_structured_operator(StructuredOperator* k) {
  if (!k) return;
  Py_XDECREF(k->points);
  Py_XDECREF(k->mat);
  Py_XDECREF(k->rhs_list);
  Py_XDECREF(k->pyobj);
  free(k);
  if (Py_IsInitialized()) Py_Finalize();
}