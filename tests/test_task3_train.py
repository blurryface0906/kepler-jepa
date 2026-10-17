"""Engineering checks; tiny CPU runs are explicitly marked smoke tests."""

import copy
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.models import M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg
from src.train import TrainConfig, batch_variance_loss, train_model, training_losses


# Task 3's archived objective is exercised with its original architectures.
# The separate Task 3.1 tests exercise the new odd/Koopman defaults.
def _task3_constructor(kind):
    kwargs = {"mlp_variant": "legacy_gelu"}
    if kind is not M0_RandomJEPA:
        kwargs["predictor_type"] = "legacy_mlp"
    constructor = partial(kind, **kwargs)
    constructor.__name__ = kind.__name__
    return constructor


M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg = (
    _task3_constructor(kind) for kind in
    (M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg))


class _IdentityPredictor(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.H_max = horizon

    def forward(self, z, k=1):
        return z


def _small_config(**changes):
    values = dict(steps=3, batch_size=4, context_length=2, delta=2, horizon=3,
                  ar_steps=2, device="cpu", smoke_test=True,
                  log_every=1, checkpoint_every=1)
    values.update(changes)
    return TrainConfig(**values)


class TrainConfigTests(unittest.TestCase):
    def test_production_protocol_cannot_silently_run_short_or_on_cpu(self):
        config = TrainConfig()
        self.assertEqual((config.steps, config.batch_size, config.horizon, config.ar_steps),
                         (2000, 256, 64, 16))
        for change in ({"device": "cpu"}, {"steps": 20}, {"batch_size": 8},
                       {"horizon": 32}, {"ar_steps": 2}, {"delta": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TrainConfig(**change)

    def test_config_rejects_invalid_optimization_values(self):
        for change in ({"steps": True}, {"batch_size": 1}, {"ar_steps": 8},
                       {"learning_rate": float("nan")}, {"gradient_clip": 0},
                       {"ema_start": .9999}, {"weight_decay": -1},
                       {"minimum_learning_rate": 1}, {"smoke_test": "yes"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _small_config(**change)

    def test_gpu_unavailable_does_not_fall_back(self):
        model = M1_IJEPA1D(L=16, C=2, d=4)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "without CPU fallback"):
                train_model(model, np.zeros((2, 1024, 2)), TrainConfig(), 42)


class TrainingObjectiveTests(unittest.TestCase):
    def test_direct_and_rollout_velocity_use_target_at_zero(self):
        config = _small_config(context_length=1, horizon=2, ar_steps=2,
                               variance_weight=0)
        model = M1_IJEPA1D(L=1, C=2, d=2, H_max=2, arch_type="linear",
                          dtype=torch.float64)
        with torch.no_grad():
            for encoder in (model.context_encoder, model.target_encoder):
                encoder.network.weight.zero_()
                encoder.network.weight[0, 0] = 1
                encoder.network.bias.zero_()
            model.target_encoder.network.bias[0] = 2
        model.predictor = _IdentityPredictor(2)
        windows = torch.zeros(4, 3, 1, 2, dtype=torch.float64)
        windows[:, :, 0, 0] = torch.tensor([0., 1., 3.], dtype=torch.float64)
        loss = training_losses(model, windows, config)
        self.assertAlmostEqual(float(loss["direct"].detach()), 17)
        self.assertAlmostEqual(float(loss["ar_state"].detach()), 17)
        self.assertAlmostEqual(float(loss["ar_velocity"].detach()), 2.5)
        self.assertAlmostEqual(float(loss["total"].detach()), 36.5)
        loss["total"].backward()
        self.assertTrue(any(p.grad is not None for p in model.context_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_batch_barrier_population_formula_and_online_gradients(self):
        config = _small_config()
        z = torch.tensor([[.1, .03], [.2, .05], [.3, .02], [.4, .04]], requires_grad=True)
        dz = torch.tensor([[.001, .002], [.003, .002], [.004, .003], [.002, .001]],
                          requires_grad=True)
        result = batch_variance_loss(z, dz, config)
        expected = ((.5 - (z.var(0, correction=0) + 1e-6).sqrt()).clamp_min(0).mean()
                    + (.05 - (dz.var(0, correction=0) + 1e-6).sqrt()).clamp_min(0).mean())
        torch.testing.assert_close(result, expected)
        result.backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertGreater(float(dz.grad.abs().sum()), 0)

    def test_barrier_is_over_batch_and_does_not_force_equal_orbit_radii(self):
        config = _small_config()
        states = torch.tensor([[-2., -4.], [-1., -2.], [1., 2.], [2., 4.]])
        increments = states * .1
        self.assertEqual(float(batch_variance_loss(states, increments, config)), 0)

    def test_all_model_objectives_decompose_and_have_no_target_gradient(self):
        config = _small_config()
        for kind in (M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg):
            with self.subTest(model=kind.__name__):
                torch.manual_seed(2)
                model = kind(L=2, C=2, d=4, H_max=3, arch_type="linear")
                windows = torch.randn(4, 4, 2, 2)
                terms = training_losses(model, windows, config)
                expected = (terms["direct"] + terms["ar"] + .1 * terms["variance"]
                            + .2 * terms["orthogonality"] + terms["isotropic"])
                torch.testing.assert_close(terms["total"], expected)
                terms["total"].backward()
                self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
                self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
                if kind is M3_IJEPA_SIGReg:
                    self.assertEqual(float(terms["variance"]), 0)
                    self.assertGreater(float(terms["isotropic"].detach()), 0)
                elif kind is M2_JEPA_OPF:
                    self.assertEqual(float(terms["sparsity"]), 0)
                    self.assertGreater(float(terms["orthogonality"].detach()), 0)

    def test_frozen_random_model_and_incomplete_target_grid_are_rejected(self):
        config = _small_config()
        windows = torch.zeros(4, 4, 2, 2)
        with self.assertRaises(TypeError):
            training_losses(M0_RandomJEPA(L=2, C=2, d=4, H_max=3), windows, config)
        with self.assertRaises(ValueError):
            training_losses(M1_IJEPA1D(L=2, C=2, d=4, H_max=3), windows[:, 1:], config)


class TrainingCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def _model_data(self):
        torch.manual_seed(11)
        model = M1_IJEPA1D(L=2, C=2, d=4, H_max=3, arch_type="linear")
        data = np.random.default_rng(11).normal(size=(3, 20, 2)).astype(np.float32)
        return model, data

    def test_smoke_train_updates_online_and_ema_and_saves_complete_checkpoint(self):
        model, data = self._model_data()
        before = copy.deepcopy(model.state_dict())
        config = _small_config()
        records = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "checkpoint.pt"
            result = train_model(model, data, config, 42, path, records.append,
                                 provenance={"purpose": "unit test only"})
            saved = torch.load(path, weights_only=True)
        self.assertEqual(result["completed_steps"], 3)
        self.assertEqual([row["step"] for row in records], [1, 2, 3])
        self.assertAlmostEqual(records[0]["lr"], 1e-3)
        self.assertAlmostEqual(records[-1]["next_lr"], 1e-5)
        self.assertAlmostEqual(records[0]["ema_momentum"], .99)
        self.assertAlmostEqual(records[-1]["ema_momentum"], .999)
        self.assertFalse(model.training)
        self.assertFalse(model.target_encoder.training)
        self.assertFalse(torch.equal(before["context_encoder.network.weight"],
                                     model.context_encoder.network.weight))
        self.assertFalse(torch.equal(before["target_encoder.network.weight"],
                                     model.target_encoder.network.weight))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
        self.assertEqual(saved["step"], 3)
        self.assertEqual(saved["config"], result["config"])
        self.assertEqual(saved["provenance"]["caller"]["purpose"], "unit test only")
        for name in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                     "rng_state", "history", "provenance"):
            self.assertIn(name, saved)
        self.assertEqual(len(saved["provenance"]["data_sha256"]), 64)

    def test_resume_matches_uninterrupted_weights_exactly(self):
        config = _small_config(steps=4)
        model_full, data = self._model_data()
        initial = copy.deepcopy(model_full.state_dict())
        train_model(model_full, data, config, 42)
        model_partial, _ = self._model_data()
        model_partial.load_state_dict(initial)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "checkpoint.pt"

            def interrupt_after_saved_step(record):
                # Step 2 starts after the checkpoint for step 1 was persisted.
                if record["step"] == 2:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_model(model_partial, data, config, 42, path, interrupt_after_saved_step)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 1)
            model_resumed, _ = self._model_data()
            resumed = train_model(model_resumed, data, config, 42, path, resume_from=path)
        self.assertEqual(resumed["completed_steps"], 4)
        for key, value in model_full.state_dict().items():
            torch.testing.assert_close(value, model_resumed.state_dict()[key], rtol=0, atol=0)

    def test_resume_rejects_changed_data_or_protocol(self):
        model, data = self._model_data()
        config = _small_config()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "checkpoint.pt"
            train_model(model, data, config, 42, path)
            with self.assertRaisesRegex(ValueError, "data_sha256"):
                train_model(model, data + .01, config, 42, resume_from=path)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                train_model(model, data, replace(config, steps=4), 42, resume_from=path)

    def test_invalid_inputs_and_nonfinite_training_loss_fail_loudly(self):
        model, data = self._model_data()
        config = _small_config()
        for invalid in (np.zeros((3, 3, 2)), np.full((3, 20, 2), np.nan),
                        np.zeros((3, 20, 2), dtype=np.complex64)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                train_model(model, invalid, config, 42)
        with mock.patch("src.train.training_losses", return_value={"total": torch.tensor(float("nan"))}):
            with self.assertRaisesRegex(FloatingPointError, "step 1"):
                train_model(model, data, config, 42)


if __name__ == "__main__":
    unittest.main()
