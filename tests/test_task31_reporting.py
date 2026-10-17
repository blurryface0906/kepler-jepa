"""Publication gating and undefined-seed reporting are scientific guarantees."""

import unittest

from experiments.exp2_reporting import _figure_missing, _markdown, _moment


def completed_fixture():
    seeds = list(range(42, 47))
    return {
        "status": "completed", "execution_completed": True,
        "protocol": {"seeds": seeds, "smoke_test": False,
                     "training": {"batch_size": 256}},
        "training_runs": [{"seed": seed, "model": model, "status": "completed",
                           "completed_steps": 2000, "device": "cuda:0"}
                          for seed in seeds for model in ("M1", "M2", "M3")],
        "frequency_runs": [{"seed": seed, "model": model, "split": "test_id_sub",
                            "regime": regime, "method": "S3b_diff"}
                           for seed in seeds for model in ("M0", "M1", "M2")
                           for regime in ("T1", "T2", "T3", "T4", "T4_512")],
        "scale_runs": [{"seed": seed, "model": model, "split": "test_id_sub", "regime": regime}
                       for seed in seeds for model in ("M1", "M2", "M3") for regime in ("T1", "T3")],
        "portraits": [{"seed": 42, "model": model, "split": "test_id_sub"}
                      for model in ("M0", "M1", "M2")],
        "checks": [{"criterion": "negative result retained", "passed": False}],
    }


class Task31ReportingTests(unittest.TestCase):
    def test_complete_negative_result_still_qualifies_for_publication_figure(self):
        self.assertEqual(_figure_missing(completed_fixture()), [])

    def test_failed_training_is_not_counted_as_completed_or_figure_ready(self):
        report = completed_fixture()
        report["training_runs"][0]["status"] = "failed"
        missing = _figure_missing(report)
        self.assertTrue(any("14/15" in message for message in missing))
        self.assertIn("Completed training records: **14/15**", _markdown(report, missing, False))

    def test_duplicate_seed_cannot_substitute_for_missing_training(self):
        report = completed_fixture()
        report["training_runs"][0] = report["training_runs"][1].copy()
        self.assertTrue(any("unique" in message for message in _figure_missing(report)))

    def test_undefined_seed_is_not_removed_from_mean_and_sd(self):
        result = _moment([{"r2": 1.0}, {"r2": float("nan")}, {"r2": 0.9}], "r2")
        self.assertEqual(result, "undefined (1/3 seeds)")

    def test_extended_direct_window_is_required_and_named_separately(self):
        report = completed_fixture()
        report["frequency_runs"] = [record for record in report["frequency_runs"]
                                    if record["regime"] != "T4_512"]
        missing = _figure_missing(report)
        self.assertTrue(any("T4_512" in message for message in missing))
        markdown = _markdown(report, missing, False)
        self.assertIn("10 seed/horizon records", markdown)
        self.assertIn("20.44", markdown)
        self.assertIn("state-norm R²", markdown)


if __name__ == "__main__":
    unittest.main()
