"""Architectural/gradient checks, independent of scientific acceptance runs."""

import unittest
from functools import partial

import torch
from torch import nn

from src.losses import ijepa_prediction_loss, opf_orthogonality_loss as _opf_loss, sigreg_isotropic_loss


def opf_orthogonality_loss(z, K=4, eps=1e-6):
    """Historical tests retain their original Task 2 objective explicitly."""
    return _opf_loss(z, K, eps, scale_invariant=False, reduction="legacy")
from src.models import (HorizonPredictor, M0_RandomJEPA, M1_IJEPA1D,
                        M2_JEPA_OPF, M3_IJEPA_SIGReg, SequenceEncoder)


# These checks describe the archived Task 2 architectures. Explicit options
# keep their meaning when Task 3.1 changes the production defaults.
def _archive_constructor(kind):
    kwargs = {"mlp_variant": "legacy_gelu"}
    if kind not in (SequenceEncoder, M0_RandomJEPA):
        kwargs["predictor_type"] = "legacy_mlp"
    constructor = partial(kind, **kwargs)
    constructor.__name__ = kind.__name__
    return constructor


SequenceEncoder, M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg = (
    _archive_constructor(kind) for kind in
    (SequenceEncoder, M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny unit-test matrices are slower with a large CPU thread pool.
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(1234)

    def test_encoder_shapes_dtypes_and_parameter_gradients(self):
        for arch in ("linear", "mlp", "transformer"):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(arch=arch, dtype=dtype):
                    encoder = SequenceEncoder(L=5, C=2, d=12, arch_type=arch, dtype=dtype)
                    x = torch.randn(3, 5, 2, dtype=dtype, requires_grad=True)
                    output = encoder(x)
                    self.assertEqual(output.shape, (3, 12))
                    self.assertEqual(output.dtype, dtype)
                    output.square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.norm().item(), 0)
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in encoder.parameters()))

    def test_linear_initialization_is_orthogonal_in_native_double_precision(self):
        encoder = SequenceEncoder(dtype=torch.float64)
        weight = encoder.network.weight
        identity = torch.eye(64, dtype=torch.float64)
        torch.testing.assert_close(weight.T @ weight, identity, rtol=0, atol=1.6e-15)
        self.assertTrue(torch.all(encoder.network.bias != 0))
        # An orthogonal map preserves the metric of the 2D harmonic subspace,
        # including highly unequal position/velocity amplitudes in the input.
        t = torch.arange(16, dtype=torch.float64) * 0.017
        cosine = torch.stack((2 * t.cos(), 2 * t.sin(), -0.7 * t.sin(), 0.7 * t.cos()), dim=1)
        sine = torch.stack((-2 * t.sin(), 2 * t.cos(), -0.7 * t.cos(), -0.7 * t.sin()), dim=1)
        basis = torch.stack((cosine.flatten(), sine.flatten()), dim=1)
        projected = weight @ basis
        torch.testing.assert_close(projected.T @ projected, basis.T @ basis, rtol=8e-16, atol=3e-14)

    def test_all_linear_layers_are_orthogonal_with_nonzero_biases(self):
        for arch in ("linear", "mlp", "transformer"):
            model = M0_RandomJEPA(L=3, C=2, d=8, H_max=5, arch_type=arch, dtype=torch.float64)
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    weight = layer.weight
                    gram = weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
                    torch.testing.assert_close(gram, torch.eye(gram.shape[0], dtype=gram.dtype),
                                               rtol=0, atol=2e-15)
                    self.assertTrue(torch.all(layer.bias != 0))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_predictor_horizon_batch_equivalence_and_residual_identity(self):
        predictor = HorizonPredictor(d=6, H_max=4, dtype=torch.float64)
        z = torch.randn(3, 6, dtype=torch.float64)
        batch_horizons = torch.tensor([1, 3, 4])
        batched = predictor(z, batch_horizons)
        individual = torch.cat([predictor(z[i:i + 1], int(k)) for i, k in enumerate(batch_horizons)])
        torch.testing.assert_close(batched, individual)
        torch.testing.assert_close(predictor(z, torch.tensor(2)), predictor(z, 2))
        for p in predictor.mlp.parameters():
            nn.init.zeros_(p)
        torch.testing.assert_close(predictor(z, batch_horizons), z, rtol=0, atol=0)

    def test_models_share_sequence_and_rollout_api(self):
        for model_class in (M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg):
            with self.subTest(model=model_class.__name__):
                model = model_class(L=3, C=2, d=8, H_max=5, arch_type="linear", dtype=torch.float64)
                model.eval()
                windows = torch.randn(7, 3, 2, dtype=torch.float64)
                torch.testing.assert_close(model.encode_sequence(windows, batch_size=3), model.encoder(windows))
                batched_windows = windows.expand(2, -1, -1, -1)
                self.assertEqual(model.encode_sequence(batched_windows).shape, (2, 7, 8))
                z0 = model.encoder(windows[:2])
                autoregressive = model.rollout_autoregressive(z0, 3)
                direct = model.rollout_direct(z0, 3)
                self.assertEqual(autoregressive.shape, (2, 3, 8))
                self.assertEqual(direct.shape, (2, 3, 8))
                torch.testing.assert_close(autoregressive[:, 0], model.predictor(z0, 1))
                torch.testing.assert_close(autoregressive[:, 1], model.predictor(autoregressive[:, 0], 1))
                torch.testing.assert_close(direct[:, 2], model.predictor(z0, 3))
                self.assertEqual(model.rollout_autoregressive(z0, 0).shape, (2, 0, 8))
                self.assertEqual(model.rollout_direct(z0, 0).shape, (2, 0, 8))
                # Autoregression uses m1 and can extend beyond the embedding table.
                self.assertEqual(model.rollout_autoregressive(z0, 6).shape, (2, 6, 8))
                with self.assertRaises(ValueError):
                    model.rollout_direct(z0, 6)

    def test_target_copy_is_frozen_and_ema_updates_parameters_and_buffers(self):
        model = M1_IJEPA1D(L=3, C=2, d=8, H_max=4, arch_type="transformer", dtype=torch.float64)
        model.train()
        self.assertTrue(model.context_encoder.training)
        self.assertFalse(model.target_encoder.training)
        self.assertFalse(any(p.requires_grad for p in model.target_encoder.parameters()))
        for context, target in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
            self.assertNotEqual(context.data_ptr(), target.data_ptr())
            torch.testing.assert_close(context, target, rtol=0, atol=0)
        # Extra buffers exercise both floating EMA and integer-copy semantics.
        model.context_encoder.register_buffer("counter", torch.tensor(7))
        model.target_encoder.register_buffer("counter", torch.tensor(2))
        before = [p.clone() for p in model.target_encoder.parameters()]
        before_buffer = model.target_encoder.position_encoding.clone()
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(0.4)
            model.context_encoder.position_encoding.add_(0.8)
        model.update_target_ema(0.75)
        for old, context, target in zip(before, model.context_encoder.parameters(), model.target_encoder.parameters()):
            torch.testing.assert_close(target, old * 0.75 + context * 0.25)
            self.assertIsNone(target.grad)
        torch.testing.assert_close(model.target_encoder.position_encoding, before_buffer + 0.2)
        self.assertEqual(model.target_encoder.counter.item(), 7)

    def test_training_loss_and_rollout_preserve_context_predictor_gradients(self):
        model = M1_IJEPA1D(L=3, C=2, d=8, H_max=4, dtype=torch.float64)
        context = torch.randn(5, 3, 2, dtype=torch.float64)
        target = torch.randn_like(context, requires_grad=True)
        loss = model.training_loss(context, target, k=torch.tensor([1, 2, 3, 4, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.context_encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
        self.assertIsNone(target.grad)
        initial = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        model.rollout_autoregressive(initial, 3).square().sum().backward()
        self.assertGreater(initial.grad.norm().item(), 0)

    def test_factor_predictors_are_independent_and_respect_factor_boundaries(self):
        model = M2_JEPA_OPF(L=3, C=2, d=12, K=3, H_max=4, dtype=torch.float64)
        factor_parameters = [{p.data_ptr() for p in predictor.parameters()} for predictor in model.predictors]
        for j in range(3):
            for k in range(j):
                self.assertFalse(factor_parameters[j] & factor_parameters[k])
        z = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
        output = model.predictor(z, 3)
        changed = z.detach().clone()
        changed[:, :4] += 2
        changed_output = model.predictor(changed, 3)
        self.assertGreater((output[:, :4] - changed_output[:, :4]).norm().item(), 0)
        torch.testing.assert_close(output[:, 4:], changed_output[:, 4:], rtol=0, atol=0)
        output[:, :4].sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.predictors[0].parameters()))
        # Concatenation creates zero gradients for the unused factors, not shared
        # parameter dependence on the output factor being differentiated.
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                            for predictor in model.predictors[1:] for p in predictor.parameters()))
        torch.testing.assert_close(z.grad[:, 4:], torch.zeros_like(z.grad[:, 4:]), rtol=0, atol=0)

    def test_active_factor_uses_temporal_variance_even_for_shifted_orbits(self):
        model = M2_JEPA_OPF(L=3, C=2, d=12, K=3)
        time = torch.linspace(-1, 1, 20)
        sequence = torch.zeros(20, 12)
        sequence[:, 4:8] = 3 * time[:, None]
        sequence[:, 8:] = time[:, None]
        self.assertEqual(model.get_active_factor(sequence), 1)
        batch = sequence.expand(2, -1, -1).clone()
        batch[1, :, :4] = 1000  # Constant inter-orbit offset must not become temporal variation.
        self.assertEqual(model.get_active_factor(batch), 1)
        self.assertEqual(model.get_active_factor(torch.zeros(20, 12)), 0)

    def test_regularized_models_objective_adds_requested_penalty(self):
        pred = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(pred, requires_grad=True)
        context = torch.randn_like(pred, requires_grad=True)
        isotropic = M3_IJEPA_SIGReg(L=3, C=2, d=8, dtype=torch.float64)
        loss = isotropic.compute_loss(pred, target, z_context=context, sigreg_weight=0.3)
        expected = ijepa_prediction_loss(pred, target) + 0.3 * sigreg_isotropic_loss(context)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertGreater(context.grad.norm().item(), 0)
        self.assertIsNone(target.grad)
        torch.testing.assert_close(isotropic.compute_loss(pred, target, sigreg_weight=0),
                                   ijepa_prediction_loss(pred, target), rtol=0, atol=0)
        factorized = M2_JEPA_OPF(L=3, C=2, d=8, dtype=torch.float64)
        torch.testing.assert_close(
            factorized.compute_loss(pred, target, z_context=context, orthogonality_weight=0.7),
            ijepa_prediction_loss(pred, target) + 0.7 * opf_orthogonality_loss(context, 4))

    def test_models_reject_invalid_configuration_shapes_and_horizons(self):
        cases = [lambda: SequenceEncoder(L=True), lambda: SequenceEncoder(C=0),
                 lambda: SequenceEncoder(d=3, arch_type="transformer"),
                 lambda: SequenceEncoder(arch_type="unknown"),
                 lambda: SequenceEncoder(dtype=torch.float16),
                 lambda: HorizonPredictor(H_max=0), lambda: M2_JEPA_OPF(d=10, K=4),
                 lambda: M2_JEPA_OPF(K=True)]
        predictor = HorizonPredictor(d=4, H_max=3)
        z = torch.ones(2, 4)
        for k in (0, 4, True, 1.0, torch.tensor([1]), torch.tensor([1.0, 2.0]),
                  torch.tensor([1, 4]), torch.ones(2, 1, dtype=torch.long)):
            cases.append(lambda k=k: predictor(z, k))
        model = M1_IJEPA1D(L=3, C=2, d=8)
        cases.extend((lambda: model.encoder(torch.zeros(2, 3, 4)),
                      lambda: model.encode_sequence(torch.zeros(3, 2)),
                      lambda: model.rollout_direct(torch.zeros(2, 4), 3),
                      lambda: model.rollout_autoregressive(torch.zeros(2, 8), -1),
                      lambda: model.update_target_ema(1.1),
                      lambda: model.update_target_ema(float("nan")),
                      lambda: M3_IJEPA_SIGReg().compute_loss(z, z, sigreg_weight=-1)))
        for index, case in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                case()


class LossTests(unittest.TestCase):
    def test_prediction_mse_detaches_target(self):
        pred = torch.tensor([[1., 3.], [4., 2.]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[2., 1.], [4., 0.]], dtype=torch.float64, requires_grad=True)
        loss = ijepa_prediction_loss(pred, target)
        self.assertEqual(loss.item(), 2.25)
        loss.backward()
        torch.testing.assert_close(pred.grad, 2 * (pred.detach() - target.detach()) / pred.numel())
        self.assertIsNone(target.grad)

    def test_orthogonality_matches_explicit_ordered_pair_formula(self):
        torch.manual_seed(6)
        z = torch.randn(11, 12, dtype=torch.float64, requires_grad=True)
        parts = z.reshape(11, 3, 4)
        standardized = (parts - parts.mean(dim=0)) / parts.std(dim=0, correction=1)
        expected = sum(((standardized[:, j].T @ standardized[:, k] / 10).square().sum())
                       for j in range(3) for k in range(3) if j != k) / 6
        actual = opf_orthogonality_loss(z, K=3)
        torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)
        actual.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.norm().item(), 0)

    def test_orthogonality_ignores_offsets_scales_and_constant_coordinates(self):
        t = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64)
        independent = torch.tensor([-1., 1., -1., 1.], dtype=torch.float64)
        z = torch.stack((t, independent), dim=1)
        self.assertLess(opf_orthogonality_loss(z, K=2).item(), 1e-30)
        duplicated = torch.stack((t, t), dim=1)
        torch.testing.assert_close(opf_orthogonality_loss(duplicated, K=2), torch.tensor(1., dtype=z.dtype))
        transformed = duplicated * torch.tensor([2., -3.]) + torch.tensor([20., 4.])
        torch.testing.assert_close(opf_orthogonality_loss(transformed, K=2),
                                   opf_orthogonality_loss(duplicated, K=2))
        constant = torch.ones(4, 8, dtype=torch.float64, requires_grad=True)
        loss = opf_orthogonality_loss(constant)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(constant.grad).all())
        mixed = torch.cat((duplicated, torch.ones_like(duplicated)), dim=1).requires_grad_()
        opf_orthogonality_loss(mixed, K=2).backward()
        self.assertTrue(torch.isfinite(mixed.grad).all())

    def test_isotropic_loss_matches_unbiased_covariance_formula_and_has_gradients(self):
        z = torch.tensor([[1., 2.], [2., -1.], [-3., 0.]], dtype=torch.float64, requires_grad=True)
        mean = z.mean(dim=0)
        covariance = (z - mean).T @ (z - mean) / 2
        expected = mean.square().sum() + (covariance - torch.eye(2)).square().sum()
        loss = sigreg_isotropic_loss(z)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.norm().item(), 0)
        # A symmetric orthogonal sample has exactly mean zero and sample Cov I.
        root = torch.sqrt(torch.tensor(1.5, dtype=torch.float64))
        white = root * torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float64)
        self.assertLess(sigreg_isotropic_loss(white).item(), 1e-29)

    def test_losses_reject_invalid_inputs(self):
        z = torch.ones(4, 8)
        cases = [lambda: ijepa_prediction_loss(z, z[:, :1]),
                 lambda: ijepa_prediction_loss(z, z.double()),
                 lambda: ijepa_prediction_loss(torch.empty(0, 8), torch.empty(0, 8)),
                 lambda: opf_orthogonality_loss(z, K=1),
                 lambda: opf_orthogonality_loss(z, K=3),
                 lambda: opf_orthogonality_loss(z, K=True),
                 lambda: opf_orthogonality_loss(z, eps=0),
                 lambda: opf_orthogonality_loss(z, eps=float("nan")),
                 lambda: opf_orthogonality_loss(z, eps=True)]
        for loss in (opf_orthogonality_loss, sigreg_isotropic_loss):
            for invalid in (torch.ones(1, 8), torch.ones(2, 0), torch.ones(2, 3, 4),
                            torch.ones(4, 8, dtype=torch.int64), torch.ones(4, 8, dtype=torch.complex64)):
                cases.append(lambda loss=loss, invalid=invalid: loss(invalid))
        for index, case in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                case()


if __name__ == "__main__":
    unittest.main()
