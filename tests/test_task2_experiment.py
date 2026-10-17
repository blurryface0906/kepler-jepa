"""Experiment acceptance must not turn invalid numerical scores into success."""

import copy
import math
import unittest

from experiments.exp1_encoder_tautology import ENCODERS, LENGTHS, METHODS, SEEDS, _criteria


class AcceptanceTests(unittest.TestCase):
    @staticmethod
    def valid_records():
        return [{"encoder": encoder, "T_eval": length, "method": method, "seed": seed,
                 "invalid": 0, "slope_abs_error": 1e-14,
                 "one_minus_r2": .011 if method == "S1" else 1e-25}
                for encoder in ENCODERS for length in LENGTHS for method in METHODS for seed in SEEDS]

    def test_missing_or_invalid_continuous_scores_cannot_pass_h2(self):
        for ratio in (float("nan"), float("inf"), -1.0):
            records = self.valid_records()
            for record in records:
                if (record["encoder"], record["T_eval"], record["method"], record["seed"]) == (
                        "M0-Linear", 4096, "S3a_diff", 42):
                    record["one_minus_r2"] = ratio
            with self.subTest(ratio=ratio):
                self.assertFalse(_criteria(records)[-1]["passed"])

    def test_exact_zero_residual_is_valid_perfect_control(self):
        records = self.valid_records()
        for record in records:
            if record["method"] in ("S3a_diff", "S3b_diff"):
                record["one_minus_r2"] = 0.0
        checks = _criteria(records)
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertTrue(all(math.isinf(v) for values in checks[-1]["improvement_decades"].values() for v in values))

    def test_worst_seed_above_task3_bound_is_not_hidden_by_valid_mean(self):
        records = self.valid_records()
        for record in records:
            if record["encoder"] == "M0-MLP" and record["seed"] == 42 and record["T_eval"] == 4096:
                if record["method"] in ("S3b_std", "S3b_diff"):
                    record["one_minus_r2"] = 1.6e-5
        untouched = copy.deepcopy(records)
        check = next(c for c in _criteria(records) if c["criterion"].startswith("H1 M0-MLP"))
        self.assertFalse(check["passed"])
        self.assertEqual(records, untouched)


if __name__ == "__main__":
    unittest.main()
