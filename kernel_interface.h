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
    PyArrayObject *mat;
    int n_points;
    PyObject *pyobj;
} Kernel;

Kernel* initialize_kernel(const char *class_name, double arg1, double kappa);
int mv_kernel_real(Kernel *kernel, const double *input, double *output, int len);
int mv_kernel_complex(Kernel *k, const double _Complex *input, double _Complex *output, int len);
const double* get_points(Kernel *k);
size_t get_n_points(Kernel *k);
void finalize_kernel(Kernel *kernel);

#ifdef __cplusplus
}
#endif

#endif
