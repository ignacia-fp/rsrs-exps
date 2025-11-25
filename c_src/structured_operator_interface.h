#ifndef STRUCTURED_OPERATOR_INTERFACE_H
#define STRUCTURED_OPERATOR_INTERFACE_H

#include <Python.h>
#include <complex.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  PyArrayObject* points;  // N x 3 array of geometry points
  PyObject* rhs_list;     // Python list of NumPy arrays (always a list)
  PyArrayObject* mat;     // operator matrix (optional)
  int n_points;           // number of points
  int n_rhs;              // number of RHS arrays (1 or 10)
  PyObject* pyobj;        // Python object instance
} StructuredOperator;

/* Initialize Python and construct the operator */
StructuredOperator* initialize_structured_operator(
    const char* python_executable, const char* class_name, double arg1,
    const char* geometry_type, double kappa, const char* precision,
    int n_sources);

/* Matrix-vector multiplication */
int mv_structured_operator_real(StructuredOperator* k, const double* input,
                                double* output, int len);
int mv_structured_operator_complex(StructuredOperator* k,
                                   const double _Complex* input,
                                   double _Complex* output, int len);

// New: single-precision real
int mv_structured_operator_real32(StructuredOperator* k, const float* input,
                                  float* output, int len);

// New: single-precision complex
int mv_structured_operator_complex32(StructuredOperator* k,
                                     const float _Complex* input,
                                     float _Complex* output, int len);

// New: get all RHS (real, float32)
const float** get_all_real_rhs_f32(StructuredOperator* k, int* n_rhs,
                                   int* len_out);

// New: get all RHS (complex, complex32)
const float _Complex** get_all_complex_rhs_f32(StructuredOperator* k,
                                               int* n_rhs, int* len_out);

/* Retrieve geometry points */
const double* get_points(StructuredOperator* k);
size_t get_n_points(StructuredOperator* k);

/* Retrieve condition number */
double get_condition_number(StructuredOperator* k);

/* Retrieve all RHS arrays at once (real or complex)
   Returns a malloc'd array of pointers. Each inner pointer points to NumPy data
   (do not free). Caller must free the outer array.
*/
const double** get_all_real_rhs(StructuredOperator* k, int* n_rhs,
                                int* len_out);
const double _Complex** get_all_complex_rhs(StructuredOperator* k, int* n_rhs,
                                            int* len_out);

/* Clean up all Python/C resources */
void finalize_structured_operator(StructuredOperator* k);

#ifdef __cplusplus
}
#endif

#endif
