#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyArrayObject *points;
    PyArrayObject *rhs;
    PyArrayObject *mat;
    int n_points;
    PyObject *pyobj;
} StructuredOperator;

StructuredOperator* initialize_structured_operator(const char* python_executable, const char *class_name, double arg1, const char *geometry_type, double kappa);
int mv_structured_operator_real(StructuredOperator *structured_operator, const double *input, double *output, int len);
int mv_structured_operator_complex(StructuredOperator *k, const double _Complex *input, double _Complex *output, int len);
const double* get_points(StructuredOperator *k);
double get_condition_number(StructuredOperator *k);
const double * structured_operator_get_real_rhs(StructuredOperator *k);
const double _Complex* structured_operator_get_complex_rhs(StructuredOperator *k);
size_t get_n_points(StructuredOperator *k);
void finalize_structured_operator(StructuredOperator *structured_operator);

#ifdef __cplusplus
}
#endif

#endif
