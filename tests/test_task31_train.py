"""Task 3.1 objective and checkpoint checks; tiny CPU runs are engineering only."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.models import (ActionAngleKoopmanPredictor, M1_IJEPA1D, M2_JEPA_OPF,
                        M3_IJEPA_SIGReg)
from src.train import (TrainConfig, relative_prediction_loss, temporal_displacement_loss,
                       train_model, training_losses)


def _config(**changes):
    values = dict(protocol="task3_1", steps=3, batch_size=4, context_length=2,
                  delta=4, horizon=16, ar_steps=2, device="cpu", smoke_test=True,
                  log_every=1, checkpoint_every=1)
    values.update(changes)
    return TrainConfig(**values)


class TemporalObjectiveTests(unittest.TestCase):
    def test_exact_per_orbit_rms_formula_and_online_gradients(self):
        config = _config()
        x0 = torch.tensor([[[1., 2.], [3., 4.]], [[2., 0.], [0., 2.]]],
                          dtype=torch.float64, requires_grad=True)
        x_lag = (-x0.detach()).requires_grad_()
        z0 = torch.tensor([[.1, .2, .3], [.1, .1, .2]],
                          dtype=torch.float64, requires_grad=True)
        z_lag = (-z0.detach()).requires_grad_()
        result = temporal_displacement_loss(x0, x_lag, z0, z_lag, config)
        rms = lambda value: value.flatten(1).square().mean(1).sqrt()
        expected = ((.8 * rms(x0 - x_lag) - rms(z0 - z_lag)).clamp_min(0)
                    + .2 * (rms(z0) - rms(x0)).square()).mean()
        torch.testing.assert_close(result, expected)
        result.backward()
        self.assertIsNone(x0.grad)
        self.assertIsNone(x_lag.grad)
        self.assertGreater(float(z0.grad.abs().sum()), 0)
        self.assertGreater(float(z_lag.grad.abs().sum()), 0)

    def test_displacement_is_intra_orbit_and_input_scale_matched(self):
        x0 = torch.tensor([[[1., 0.]], [[5., 0.]]])
        x_lag = -x0
        # Different radii do not receive a cross-orbit penalty. Equal coordinate
        # RMS and sufficient within-orbit displacement give exactly zero loss.
        z0 = x0.flatten(1).repeat(1, 2)
        z_lag = -z0
        self.assertEqual(float(temporal_displacement_loss(x0, x_lag, z0, z_lag, _config())), 0)
        collapsed = z0.clone()
        self.assertGreater(float(temporal_displacement_loss(
            x0, x_lag, collapsed, collapsed, _config())), 0)

    def test_zero_embeddings_have_finite_subgradient_and_positive_penalty(self):
        x = torch.ones(2, 2, 2)
        z0 = torch.zeros(2, 4, requires_grad=True)
        z_lag = torch.zeros(2, 4, requires_grad=True)
        result = temporal_displacement_loss(x, -x, z0, z_lag, _config())
        self.assertAlmostEqual(float(result.detach()), 1.8, places=6)
        result.backward()
        self.assertTrue(torch.isfinite(z0.grad).all())
        self.assertTrue(torch.isfinite(z_lag.grad).all())

    def test_temporal_argument_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            temporal_displacement_loss(torch.ones(2, 2, 2), torch.ones(2, 3, 2),
                                       torch.ones(2, 4), torch.ones(2, 4), _config())


class RelativePredictionTests(unittest.TestCase):
    def test_exact_ratio_per_orbit_with_detached_targets_and_denominator(self):
        target = torch.tensor([[[1., 0.], [2., 0.]],
                               [[10., 0.], [20., 0.]]], dtype=torch.float64,
                              requires_grad=True)
        prediction = (target.detach() * torch.tensor([0., .5], dtype=torch.float64)[:, None, None])
        prediction.requires_grad_()
        result = relative_prediction_loss(prediction, target)
        expected = ((2.5 / (2.5 + 1e-4)) + (62.5 / (250 + 1e-4))) / 2
        self.assertAlmostEqual(float(result.detach()), expected, places=14)
        result.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_zero_target_is_guarded_and_shape_validation_is_explicit(self):
        prediction = torch.ones(2, 3, 4, requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=True)
        result = relative_prediction_loss(prediction, target)
        self.assertAlmostEqual(float(result.detach()), 40000, places=1)
        result.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        with self.assertRaises(ValueError):
            relative_prediction_loss(prediction[:, 0], target[:, 0])
        with self.assertRaises(ValueError):
            relative_prediction_loss(prediction, target, 0)


class _IdentityKoopman(ActionAngleKoopmanPredictor):
    """Deterministic test double with the same shared-basis call protocol."""
    def __init__(self, d, horizon):
        nn.Module.__init__(self)
        self.d, self.H_max = d, horizon
        self.basis_calls = 0

    def orthogonal_basis(self):
        self.basis_calls += 1
        return torch.eye(self.d, dtype=torch.float64)

    def forward(self, z, k=1, *, basis=None):
        return z

    def predict_horizons(self, z, horizons, *, basis=None):
        return z[:, None].expand(-1, len(horizons), -1)


class RemediationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_remediation_config_requires_fixed_production_lag_and_stride(self):
        self.assertEqual(TrainConfig().protocol, "task3")
        self.assertEqual(TrainConfig(protocol="task3_1").temporal_lag, 16)
        for change in ({"delta": 2}, {"temporal_lag": 8}, {"protocol": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                values = {"protocol": "task3_1", **change}
                TrainConfig(**values)
        with self.assertRaisesRegex(ValueError, "temporal_lag"):
            _config(horizon=8)

    def test_direct_is_relative_but_ar_keeps_original_state_and_velocity_units(self):
        config = _config(batch_size=4, context_length=1, horizon=2, ar_steps=2,
                         temporal_lag=2, temporal_weight=0)
        model = M1_IJEPA1D(L=1, C=2, d=2, H_max=2, arch_type="linear",
                          dtype=torch.float64)
        with torch.no_grad():
            for encoder in (model.context_encoder, model.target_encoder):
                encoder.network.weight.zero_()
                encoder.network.weight[0, 0] = 1
                encoder.network.bias.zero_()
            model.target_encoder.network.bias[0] = 2
        model.predictor = _IdentityKoopman(2, 2)
        windows = torch.zeros(4, 3, 1, 2, dtype=torch.float64)
        windows[:, :, 0, 0] = torch.tensor([0., 1., 3.], dtype=torch.float64)
        terms = training_losses(model, windows, config)
        self.assertAlmostEqual(float(terms["direct"].detach()), 17 / (17 + 1e-4))
        self.assertAlmostEqual(float(terms["ar_state"].detach()), 17)
        self.assertAlmostEqual(float(terms["ar_velocity"].detach()), 2.5)
        self.assertAlmostEqual(float(terms["total"].detach()), 17 / (17 + 1e-4) + 19.5)
        self.assertEqual(model.predictor.basis_calls, 1)

    def test_model_branch_objectives_and_one_basis_per_graph(self):
        config = _config()
        for kind in (M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg):
            with self.subTest(model=kind.__name__):
                torch.manual_seed(9)
                options = {"normalize_embeddings": True} if kind is M3_IJEPA_SIGReg else {}
                model = kind(L=2, C=2, d=8, H_max=16, arch_type="mlp", **options)
                windows = torch.randn(4, 17, 2, 2)
                with mock.patch.object(model.predictor, "orthogonal_basis",
                                       wraps=model.predictor.orthogonal_basis) as basis_call:
                    with mock.patch("src.train.batch_variance_loss",
                                    side_effect=AssertionError("archived batch barrier used")):
                        terms = training_losses(model, windows, config)
                self.assertEqual(basis_call.call_count, 1)
                expected = (terms["direct"] + terms["ar"] + terms["temporal"]
                            + .2 * terms["orthogonality"] + terms["isotropic"])
                torch.testing.assert_close(terms["total"], expected)
                self.assertEqual(float(terms["variance"]), 0)
                if kind is M3_IJEPA_SIGReg:
                    self.assertEqual(float(terms["temporal"]), 0)
                    self.assertGreater(float(terms["isotropic"].detach()), 0)
                else:
                    self.assertEqual(float(terms["isotropic"]), 0)
                    self.assertGreater(float(terms["temporal"].detach()), 0)
                terms["total"].backward()
                self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
                self.assertTrue(all(parameter.grad is None
                                    for parameter in model.target_encoder.parameters()))
                for module in (model.context_encoder, model.predictor):
                    gradients = [parameter.grad for parameter in module.parameters()
                                 if parameter.requires_grad and parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def _model_data(self):
        torch.manual_seed(17)
        model = M1_IJEPA1D(L=2, C=2, d=8, H_max=16, arch_type="mlp")
        data = np.random.default_rng(17).normal(size=(3, 80, 2)).astype(np.float32)
        return model, data

    def test_remediation_checkpoint_resume_is_exact_and_records_protocol_architecture(self):
        config = _config(steps=4)
        full, data = self._model_data()
        initial = copy.deepcopy(full.state_dict())
        train_model(full, data, config, 42)
        partial, _ = self._model_data()
        partial.load_state_dict(initial)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "task31.pt"

            def interrupt(record):
                if record["step"] == 2:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_model(partial, data, config, 42, checkpoint, interrupt)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["config"]["protocol"], "task3_1")
            self.assertEqual(saved["provenance"]["objective_protocol"], "task3_1")
            self.assertIn("predictor_class", saved["provenance"]["model_config"])
            self.assertIn("mlp_variant", saved["provenance"]["model_config"])
            resumed, _ = self._model_data()
            result = train_model(resumed, data, config, 42, checkpoint, resume_from=checkpoint)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                train_model(resumed, data, replace(config, protocol="task3"), 42,
                            resume_from=checkpoint)
        self.assertEqual(result["completed_steps"], 4)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
