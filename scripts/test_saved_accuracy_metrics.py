#!/usr/bin/env python3
"""Regression tests for headline normalization and non-disruptive backfills."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from saved_accuracy_metrics import accuracy_metrics, atomic_json, backfill_queue, read_json


class AccuracyMetricTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(dim=25, norm_apply_fro=4.0, norm_a_fro=20.0,
                        norm_apply_2=3.0, norm_a_2=12.0,
                        err_solve_fro=0.5, err_solve_2=0.3, solve_error_rhs=0.07,
                        tot_num_samples=100, unrelated={"keep": True})

    def test_distinct_normalizations(self):
        metrics = accuracy_metrics(self.raw)
        self.assertEqual(metrics["compression_relerr_fro"], 0.2)
        self.assertEqual(metrics["compression_relerr_2"], 0.25)
        self.assertEqual(metrics["solve_residual_relerr_fro"], 0.1)
        self.assertEqual(metrics["solve_residual_norm_2"], 0.3)
        self.assertEqual(metrics["definitions"]["solve_residual_norm_2"], "||I - B A||_2")
        self.assertTrue(metrics["values_are_estimates"])
        self.assertFalse(metrics["certified_bounds"])
        self.assertEqual(metrics["estimation"]["frobenius_probe_count"], 20)
        self.assertEqual(metrics["estimation"]["spectral_power_steps"], 50)

    def test_small_sphere_known_values(self):
        metrics = accuracy_metrics(dict(dim=299718, norm_apply_fro=3.8743538164993744e-8,
                                        norm_a_fro=7.946847268554941e-5,
                                        norm_apply_2=2.708830381075636e-9,
                                        norm_a_2=4.214991634725559e-5,
                                        err_solve_fro=0.6011219367891096,
                                        err_solve_2=0.03273044733776161))
        self.assertAlmostEqual(metrics["compression_relerr_fro"], 4.875334438387777e-4)
        self.assertAlmostEqual(metrics["compression_relerr_2"], 6.426656600593728e-5)
        self.assertAlmostEqual(metrics["solve_residual_relerr_fro"], 1.0980096678447208e-3)
        self.assertEqual(metrics["solve_residual_norm_2"], 0.03273044733776161)

    def test_invalid_values_are_null_not_zero(self):
        metrics = accuracy_metrics(dict(self.raw, dim=0, norm_a_fro=0.0, norm_a_2=0.0, err_solve_2=math.nan))
        self.assertIsNone(metrics["compression_relerr_fro"])
        self.assertIsNone(metrics["compression_relerr_2"])
        self.assertIsNone(metrics["solve_residual_relerr_fro"])
        self.assertIsNone(metrics["solve_residual_norm_2"])
        metrics = accuracy_metrics(dict(self.raw, norm_apply_fro=-1.0, err_solve_fro=None, err_solve_2=math.inf))
        self.assertIsNone(metrics["compression_relerr_fro"])
        self.assertIsNone(metrics["solve_residual_relerr_fro"])
        self.assertIsNone(metrics["solve_residual_norm_2"])

    def test_backfill_is_idempotent_and_preserves_raw_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "sphere/n25/k20_p80_leaf120"
            result = work / "results/error_stats_2e1.json"
            result.parent.mkdir(parents=True)
            atomic_json(result, self.raw)
            run = dict(state="completed", returncode=0, n_dofs=25, rank=20, p=80,
                       leaf_size=120, stats=dict(error_stats=str(result), error_stats_data=self.raw))
            atomic_json(work / "status.json", run)
            aggregate = dict(state="running", results=[copy.deepcopy(run)])
            atomic_json(root / "status.json", aggregate)
            aggregate_before = (root / "status.json").read_bytes()
            report = backfill_queue(root)
            self.assertEqual(report["result_files_updated"], 1)
            self.assertFalse(report["errors"])
            saved = read_json(result)
            self.assertEqual({k:v for k,v in saved.items() if k != "accuracy_metrics"}, self.raw)
            self.assertEqual((root / "status.json").read_bytes(), aggregate_before)
            status = read_json(work / "status.json")
            self.assertEqual(status["accuracy_metrics"], saved["accuracy_metrics"])
            self.assertEqual(status["stats"]["error_stats_data"], saved)
            before = result.stat().st_mtime_ns
            self.assertEqual(backfill_queue(root)["result_files_updated"], 0)
            self.assertEqual(result.stat().st_mtime_ns, before)
            self.assertEqual(read_json(root / "accuracy_summary.json")["results"][0]["rank"], 20)
            aggregate["state"] = "completed"
            atomic_json(root / "status.json", aggregate)
            backfill_queue(root)
            completed = read_json(root / "status.json")["results"][0]
            self.assertEqual(completed["accuracy_metrics"], saved["accuracy_metrics"])
            self.assertEqual(completed["stats"]["error_stats_data"]["accuracy_metrics"], saved["accuracy_metrics"])

    def test_running_results_are_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "sphere/n25/k20_p80_leaf120"
            work.mkdir(parents=True)
            status = dict(state="running", n_dofs=25, rank=20)
            atomic_json(work / "status.json", status)
            before = (work / "status.json").read_bytes()
            self.assertEqual(backfill_queue(root)["completed_results"], 0)
            self.assertEqual((work / "status.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
