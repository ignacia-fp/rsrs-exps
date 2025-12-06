#!/bin/bash
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
cargo run --release '{"structured_operator_type": "BemppClCombinedHelmholtz", "precision": "Double"}' '{"id_tols": [0.01, 0.001, 0.0001, 1e-06, 1e-08], "dim_args": [{"KappaAndMeshwidth": [0.1, 0.1]}], "geometry_type": "SphereSurface", "max_tree_depth": 16, "n_sources": 1}' '{"oversampling": 8, "oversampling_diag_blocks": 16, "min_num_samples": 0, "initial_num_samples": 20, "stabilise": {"type": "False"}, "null_method": "Projection", "qr_method": "RRQR", "near_block_extraction_method": "LuLstSq", "diag_block_extraction_method": "LuLstSq", "lu_pivot_method": {"type": "Lu", "value": 0}, "diag_pivot_method": {"type": "Lu", "value": 0}, "tol_null": 1e-16, "tol_id": 0.01, "tol_ext_near": 1e-16, "tol_diag_ext": 1e-16, "min_rank": 1, "min_level": 1, "symmetric": false, "rank_picking": "Min", "fact_type": "Joint", "save_samples": false, "num_threads": 64}' '{"solve": {"False": null}, "plot": true, "dense_errors": false, "factors_cn": false, "results_output": "All"}'
