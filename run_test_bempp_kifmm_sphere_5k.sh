#!/bin/bash
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYENV_VERSION="${PYENV_VERSION:-kifmm-bempp-env}"
if command -v pyenv >/dev/null 2>&1; then
  export RSRS_EXPS_PYTHON="${RSRS_EXPS_PYTHON:-$(pyenv which python)}"
fi
export PYTHONPATH="$REPO_ROOT/external/bempp-cl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${BEMPP_KIFMM_LIBRARY:-}" ]; then
  for candidate in \
    "$REPO_ROOT/external/kifmm-for-rsrs/target/release/libkifmm.dylib" \
    "$REPO_ROOT/external/kifmm-for-rsrs/target/release/libkifmm.so" \
    "$REPO_ROOT/external/kifmm-for-rsrs/target/maturin/libkifmm.dylib" \
    "$REPO_ROOT/external/kifmm-for-rsrs/target/maturin/libkifmm.so"; do
    if [ -f "$candidate" ]; then
      export BEMPP_KIFMM_LIBRARY="$candidate"
      break
    fi
  done
fi
if [ -z "${BEMPP_KIFMM_LIBRARY:-}" ]; then
  echo "Could not find a local KiFMM shared library under external/kifmm-for-rsrs/target." >&2
  exit 1
fi
KIFMM_LIB_DIR="$(dirname "$BEMPP_KIFMM_LIBRARY")"
export DYLD_LIBRARY_PATH="$KIFMM_LIB_DIR:$KIFMM_LIB_DIR/deps${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$KIFMM_LIB_DIR:$KIFMM_LIB_DIR/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "RSRS_EXPS_PYTHON=${RSRS_EXPS_PYTHON:-<unset>}"
echo "BEMPP_KIFMM_LIBRARY=$BEMPP_KIFMM_LIBRARY"
"${RSRS_EXPS_PYTHON:-python}" - <<'PY'
import sys
import bempp_cl
print(f"Python executable: {sys.executable}")
print(f"bempp_cl module: {bempp_cl.__file__}")
PY
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
unset RAYON_NUM_THREADS
cargo run --release '{"structured_operator_type": "BemppClLaplaceSingleLayer", "precision": "Double"}' '{"id_tols": [8], "dim_args": [{"MeshWidth": 0.079}], "geometry_type": "SphereSurface", "max_tree_depth": 2, "n_sources": 1, "assembler": "KiFMM"}' '{"oversampling": 8, "oversampling_diag_blocks": 16, "min_num_samples": 0, "initial_num_samples": 0, "fixed_rank_sampling_mode": "PerLevel", "run_seed": null, "shift": {"type": "False"}, "null_method": "Projection", "qr_method": "RRQR", "near_block_extraction_method": "LuLstSq", "diag_block_extraction_method": "LuLstSq", "lu_pivot_method": {"type": "LuHybrid", "value": 0}, "diag_pivot_method": {"type": "LuHybrid", "value": 0}, "tol_null": 1e-16, "tol_id": 8, "tol_ext_near": 1e-16, "tol_diag_ext": 1e-16, "min_rank": 1, "min_level": 1, "symmetry": "Symmetric", "rank_picking": "Min", "fact_type": "Joint", "save_samples": false, "load_samples": false, "num_threads": 8, "flush_factors": false, "store_far": false, "symmetric": null}' '{"solve": {"False": null}, "plot": false, "dense_errors": false, "factors_cn": false, "results_output": "All"}'
