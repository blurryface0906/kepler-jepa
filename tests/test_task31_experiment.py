"""Task 3.1 acceptance must not hide seeds, factors, or direct horizons."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from experiments.exp2_circular_trained import (PRIMARY_METHOD, SEEDS,
    _active_factor_sequences, _model, _predictor_diagnostics, _test_batches,
    acceptance_checks)
from src.evaluation import evaluate_regimes
from src.models import ActionAngleKoopmanPredictor, HorizonPredictor, OddResidualMLP
from src.train import TrainConfig


def _passing_report():
    report = {"protocol": {"smoke_test": False, "primary_method": PRIMARY_METHOD},
              "training_runs": [], "frequency_runs": [], "scale_runs": [],
              "factor_frequency_runs": []}
    for seed in SEEDS:
        for model in ("M1", "M2", "M3"):
            report["training_runs"].append({"seed": seed, "model": model, "status": "completed",
                "completed_steps": 2000, "device": "cuda:0",
                "config": {"batch_size": 256, "protocol": "task3_1"}})
        for model in ("M0", "M1", "M2"):
            for regime in ("T1", "T3", "T4", "T4_512"):
                recovery = model != "M0" or regime == "T1"
                report["frequency_runs"].append({"seed": seed, "model": model,
                    "split": "test_id_sub", "regime": regime, "method": PRIMARY_METHOD,
                    "invalid": 0, "slope_abs_error": .001 if recovery else .7,
                    "r2": .99999 if recovery else .3})
        for model in ("M1", "M2", "M3"):
            for regime in ("T1", "T3"):
                report["scale_runs"].append({"seed": seed, "model": model,
                    "split": "test_id_sub", "regime": regime, "status": "completed",
                    "state_norm_r2": 0 if model == "M3" else .5,
                    "phase_norm_r2": 0 if model == "M3" else .99,
                    "axis_probe_r2": .99,
                    "scale_r2_cycle_radius": .99999 if model == "M3" else .1})
    return report


class Task31AcceptanceTests(unittest.TestCase):
    def test_sixteen_checks_and_h4_use_new_metrics_not_old_cycle_threshold(self):
        checks = acceptance_checks(_passing_report())
        self.assertEqual(len(checks), 16)
        self.assertTrue(all(row["passed"] for row in checks))

    def test_t4_requires_each_seed_at_both_trained_and_extended_horizons(self):
        for regime in ("T4", "T4_512"):
            report = _passing_report()
            selected = next(r for r in report["frequency_runs"]
                            if r["seed"] == 46 and r["model"] == "M1" and r["regime"] == regime)
            selected["slope_abs_error"] = .051
            check = next(row for row in acceptance_checks(report) if row["criterion"].startswith("ID T4 M1"))
            self.assertFalse(check["passed"])
            self.assertEqual(check["expected_seed_records"], 10)
        report = _passing_report()
        report["frequency_runs"] = [r for r in report["frequency_runs"] if not
            (r["seed"] == 42 and r["model"] == "M1" and r["regime"] == "T4_512")]
        self.assertFalse(next(r for r in acceptance_checks(report) if r["criterion"].startswith("ID T4 M1"))["passed"])

    def test_missing_or_duplicate_seed_cannot_pass(self):
        report = _passing_report()
        index = next(i for i, r in enumerate(report["frequency_runs"])
                     if r["seed"] == 46 and r["model"] == "M2" and r["regime"] == "T1")
        report["frequency_runs"][index] = deepcopy(next(r for r in report["frequency_runs"]
            if r["seed"] == 42 and r["model"] == "M2" and r["regime"] == "T1"))
        self.assertFalse(next(r for r in acceptance_checks(report) if r["criterion"].startswith("ID T1 M2"))["passed"])

    def test_h4_both_metrics_must_pass_for_each_model_and_regime(self):
        for model, metric, value in (("M1", "phase_norm_r2", .899),
                                     ("M2", "axis_probe_r2", .949),
                                     ("M3", "state_norm_r2", .051),
                                     ("M3", "phase_norm_r2", .051)):
            report = _passing_report()
            target = next(r for r in report["scale_runs"] if r["model"] == model
                          and r["regime"] == "T3" and r["seed"] == 44)
            target[metric] = value
            self.assertFalse(next(r for r in acceptance_checks(report)
                if r["criterion"].startswith(f"H4 ID T3 {model}"))["passed"])

    def test_full64_invalid_result_cannot_be_replaced_by_good_factor_or_other_method(self):
        report = _passing_report()
        target = next(r for r in report["frequency_runs"] if r["model"] == "M2"
                      and r["regime"] == "T3" and r["seed"] == 42)
        alternative = {**target, "method": "S3a_diff"}
        report["factor_frequency_runs"].append(deepcopy(target))
        target.update(invalid=1, slope_abs_error=np.nan, r2=np.nan)
        report["frequency_runs"].append(alternative)
        self.assertFalse(next(r for r in acceptance_checks(report)
            if r["criterion"].startswith("ID T3 M2"))["passed"])

    def test_gpu_proof_requires_all_models_seeds_protocol_and_batch(self):
        for field, value in (("device", "cpu"), ("completed_steps", 1999)):
            report = _passing_report()
            report["training_runs"][0][field] = value
            self.assertFalse(acceptance_checks(report)[0]["passed"])
        report = _passing_report()
        report["training_runs"][0]["config"]["batch_size"] = 128
        self.assertFalse(acceptance_checks(report)[0]["passed"])
        report = _passing_report()
        report["training_runs"][0] = deepcopy(report["training_runs"][1])
        self.assertFalse(acceptance_checks(report)[0]["passed"])


class Task31ExperimentEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_model_factory_has_odd_encoders_unstructured_control_and_no_metadata(self):
        for name in ("M0", "M1", "M2", "M3"):
            model = _model(name, 42, "cpu")
            base_encoder = getattr(model.encoder, "base_encoder", model.encoder)
            self.assertIsInstance(base_encoder.network, OddResidualMLP)
            self.assertEqual(next(model.parameters()).dtype, torch.float32)
            self.assertIsInstance(model.predictor, HorizonPredictor if name == "M0" else ActionAngleKoopmanPredictor)
            self.assertEqual(model.predictor.H_max, 512 if name == "M0" else 64)
            self.assertEqual(any(p.requires_grad for p in model.parameters()), name != "M0")

    def test_factor_selection_is_per_orbit_temporal_variance_and_retains_invalids(self):
        t = np.linspace(-1, 1, 20)
        values = np.zeros((3, 20, 8))
        values[0, :, 0:2] = t[:, None] * 2
        values[0, :, 2:4] = 1000  # A constant inter-factor offset cannot win.
        values[1, :, 4:6] = t[:, None] * 3
        values[2, 0, 7] = np.nan
        active, indices, scores = _active_factor_sequences(values)
        self.assertEqual(indices[:2], [0, 2])
        np.testing.assert_array_equal(active[0], values[0, :, :2])
        self.assertTrue(np.isnan(active[2]).all())
        self.assertEqual(len(scores), 3)

    def test_test_grids_remain_thirty_id_fifty_ood_and_use_2100_raw_samples(self):
        batches = _test_batches(42, 2100)
        self.assertEqual(batches["test_id_sub"]["trajectories"].shape, (30, 2100, 4))
        self.assertEqual(batches["test_ood"]["trajectories"].shape, (50, 2100, 4))
        self.assertTrue(np.all((batches["test_id_sub"]["a"] >= .5) & (batches["test_id_sub"]["a"] <= 2)))

    def test_predictor_diagnostics_measure_iterative_ar_and_intra_orbit_displacement(self):
        config = TrainConfig(steps=3, batch_size=8, horizon=4, ar_steps=2, temporal_lag=2,
                            device="cpu", smoke_test=True, protocol="task3_1")
        model = _model("M1", 42, "cpu").to(dtype=torch.float64).eval()
        batch = _test_batches(42, 80)["test_id_sub"]
        sequences = evaluate_regimes(model, batch["trajectories"], N_eval=16,
                                     H_train=4, H_extended=16)
        diagnostics = _predictor_diagnostics(model, batch, sequences, config)
        for key in ("norm_relative_drift_max", "action_log_drift_max",
                    "ar_direct_relative_error_max", "orthogonality_max_abs"):
            self.assertLess(diagnostics[key], 1e-10)
        self.assertEqual(len(diagnostics["input_temporal_rms"]), 30)
        self.assertGreater(diagnostics["temporal_displacement_ratio_median"], 0)


if __name__ == "__main__":
    unittest.main()
