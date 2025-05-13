#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyArrayObject *points;
    PyArrayObject *mat;
    int dim;
    PyObject *pyobj;
} Kernel;

Kernel* initialize_kernel(const char *class_name, int dim, double kappa);
int mv_kernel(Kernel *kernel, const double *input, double *output, int len);
const double* get_points(Kernel *k);
void finalize_kernel(Kernel *kernel);

#ifdef __cplusplus
}
#endif

#endif
