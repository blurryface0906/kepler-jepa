"""Task 3's aggregate acceptance and pointwise M3 ablation contract."""

import unittest
import numpy as np
import torch

from experiments.exp1_encoder_tautology import _criteria
from tests import test_task2_experiment as acceptance_fixtures
from src.models import M3_IJEPA_SIGReg


class Task3Tests(unittest.TestCase):
    def test_aggregate_nonlinear_bound_allows_small_single_seed_overrun(self):
        records = acceptance_fixtures.AcceptanceTests.valid_records()
        for record in records:
            if record["encoder"] == "M0-MLP" and record["T_eval"] == 4096 and record["method"] == "S3b_std":
                record["one_minus_r2"] = 1.1e-5 if record["seed"] == 42 else 6e-6
        check = next(c for c in _criteria(records) if c["criterion"].startswith("H1 M0-MLP"))
        self.assertTrue(check["by_method"]["S3b_std"])
        self.assertLess(check["aggregates"]["S3b_std"]["mean_residual_ratio"], 1e-5)

    def test_invalid_nonlinear_seed_cannot_pass_aggregates(self):
        records = acceptance_fixtures.AcceptanceTests.valid_records()
        for record in records:
            if record["encoder"] == "M0-MLP" and record["seed"] == 42:
                record["one_minus_r2"] = np.nan
        check = next(c for c in _criteria(records) if c["criterion"].startswith("H1 M0-MLP"))
        self.assertFalse(check["passed"])

    def test_m3_pointwise_unit_rms_does_not_fit_an_entire_orbit(self):
        torch.manual_seed(42)
        model = M3_IJEPA_SIGReg(L=4, C=2, d=8, normalize_embeddings=True, dtype=torch.float64,
                                mlp_variant="legacy_gelu", predictor_type="legacy_mlp").eval()
        x = torch.randn(6, 4, 2, dtype=torch.float64)
        z = model.encode_sequence(x)
        torch.testing.assert_close(z.square().sum(-1), torch.full((6,), 8., dtype=torch.float64), atol=1e-13, rtol=0)
        torch.testing.assert_close(z.mean(-1), torch.zeros(6, dtype=torch.float64), atol=1e-15, rtol=0)
        torch.testing.assert_close(model.encode_sequence(x[:1]), z[:1], atol=1e-14, rtol=1e-14)
        with torch.no_grad():
            torch.testing.assert_close(model.target_encoder(x), z)
            rollout = model.rollout_autoregressive(z, 5)
            torch.testing.assert_close(rollout.square().sum(-1), torch.full((6,5), 8., dtype=torch.float64), atol=1e-13, rtol=0)
        self.assertTrue(all(not p.requires_grad for p in model.target_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
