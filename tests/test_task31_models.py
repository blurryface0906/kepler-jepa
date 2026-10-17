"""Task 3.1 architecture invariants, without fitting a physical frequency law."""

import unittest
from unittest import mock

import torch
from torch import nn

from src.models import (ActionAngleKoopmanPredictor, HorizonPredictor,
                        M0_RandomJEPA, M1_IJEPA1D, M2_JEPA_OPF,
                        M3_IJEPA_SIGReg, OddResidualMLP, SequenceEncoder)


class Task31ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)

    def _predictor(self, K=1):
        predictor = ActionAngleKoopmanPredictor(d=8, H_max=64, K=K,
                                               dtype=torch.float64)
        with torch.no_grad():
            predictor.S.normal_(std=0.09)
        return predictor

    def test_odd_encoder_has_requested_gains_and_no_bias_parameters(self):
        encoder = SequenceEncoder(L=3, C=2, d=8, arch_type="mlp", dtype=torch.float64)
        self.assertIsInstance(encoder.network, OddResidualMLP)
        for name, gain in (("skip", 1.0), ("W1", 1.0), ("W2", 1.0), ("W3", 0.15)):
            layer = getattr(encoder.network, name)
            self.assertIsNone(layer.bias)
            w = layer.weight
            gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
            expected = torch.eye(gram.shape[0], dtype=w.dtype) * gain ** 2
            torch.testing.assert_close(gram, expected, rtol=0, atol=2e-15)
        zero = torch.zeros(2, 3, 2, dtype=torch.float64)
        torch.testing.assert_close(encoder(zero), torch.zeros(2, 8, dtype=torch.float64), rtol=0, atol=0)

    def test_oddness_survives_an_optimizer_update_and_all_branches_receive_gradients(self):
        encoder = SequenceEncoder(L=3, C=2, d=8, arch_type="mlp", dtype=torch.float64)
        x = torch.randn(7, 3, 2, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(encoder(-x), -encoder(x), rtol=0, atol=0)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.003)
        encoder(x).square().mean().backward()
        for parameter in encoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.norm().item(), 0)
        optimizer.step()
        torch.testing.assert_close(encoder(-x), -encoder(x), rtol=0, atol=0)

    def test_oddness_does_not_require_or_imply_zero_incomplete_arc_mean(self):
        encoder = SequenceEncoder(L=3, C=2, d=8, arch_type="mlp", dtype=torch.float64)
        arc = torch.randn(5, 3, 2, dtype=torch.float64) + 2
        self.assertGreater(encoder(arc).mean(0).norm().item(), 0.1)
        paired = torch.cat((arc, -arc))
        torch.testing.assert_close(encoder(paired).mean(0), torch.zeros(8, dtype=torch.float64),
                                   rtol=0, atol=4e-16)

    def test_q_is_orthogonal_with_positive_determinant_and_differentiable(self):
        predictor = self._predictor()
        Q = predictor.orthogonal_basis()
        torch.testing.assert_close(Q.T @ Q, torch.eye(8, dtype=torch.float64), rtol=0, atol=2e-15)
        torch.testing.assert_close(torch.linalg.det(Q), torch.tensor(1., dtype=torch.float64),
                                   rtol=0, atol=2e-15)
        z = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(z)
        (predictor(z, 17, basis=Q) - target).square().mean().backward()
        self.assertGreater(predictor.S.grad.abs().sum().item(), 0)
        for parameter in predictor.rate_net.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(z.grad.norm().item(), 0)

    def test_positive_action_rates_horizon_batch_and_vectorized_equivalence(self):
        predictor = self._predictor()
        z = torch.randn(4, 8, dtype=torch.float64)
        Q = predictor.orthogonal_basis()
        rates = predictor.angular_rates(z, basis=Q)
        self.assertEqual(rates.shape, (4, 4))
        self.assertTrue(torch.all(rates > 0))
        ks = torch.tensor([1, 17, 64, 512])
        batched = predictor(z, ks, basis=Q)
        individual = torch.cat([predictor(z[i:i + 1], int(k), basis=Q) for i, k in enumerate(ks)])
        torch.testing.assert_close(batched, individual, rtol=2e-14, atol=2e-14)
        all_horizons = predictor.predict_horizons(z, ks, basis=Q)
        self.assertEqual(all_horizons.shape, (4, 4, 8))
        for index, k in enumerate(ks):
            torch.testing.assert_close(all_horizons[:, index], predictor(z, int(k), basis=Q),
                                       rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(predictor(z, torch.tensor(4)), predictor(z, 4))

    def test_direct_preserves_norm_actions_and_rates_beyond_training_horizon(self):
        predictor = self._predictor()
        z = torch.randn(6, 8, dtype=torch.float64)
        Q = predictor.orthogonal_basis()
        original_actions = predictor.plane_log_actions(z, basis=Q)
        original_rates = predictor.angular_rates(z, basis=Q)
        for k in (1, 64, 512, 10000):
            future = predictor(z, k, basis=Q)
            torch.testing.assert_close(future.norm(dim=1), z.norm(dim=1), rtol=2e-15, atol=2e-15)
            torch.testing.assert_close(predictor.plane_log_actions(future, basis=Q), original_actions,
                                       rtol=3e-14, atol=3e-14)
            torch.testing.assert_close(predictor.angular_rates(future, basis=Q), original_rates,
                                       rtol=2e-14, atol=2e-14)

    def test_semigroup_matches_two_composed_horizons(self):
        predictor = self._predictor()
        z = torch.randn(5, 8, dtype=torch.float64)
        Q = predictor.orthogonal_basis()
        composed = predictor(predictor(z, 31, basis=Q), 481, basis=Q)
        torch.testing.assert_close(composed, predictor(z, 512, basis=Q), rtol=8e-13, atol=8e-13)

    def test_m2_basis_has_factor_blocks_and_each_factor_norm_is_preserved(self):
        model = M2_JEPA_OPF(L=3, C=2, d=16, K=4, dtype=torch.float64)
        with torch.no_grad():
            model.predictor.S.normal_(std=0.1)
        Q = model.predictor.orthogonal_basis()
        for row in range(4):
            for column in range(4):
                block = Q[row * 4:(row + 1) * 4, column * 4:(column + 1) * 4]
                if row != column:
                    self.assertEqual(torch.count_nonzero(block).item(), 0)
        torch.testing.assert_close(Q.T @ Q, torch.eye(16, dtype=torch.float64), rtol=0, atol=2e-15)
        z = torch.randn(5, 16, dtype=torch.float64)
        future = model.predictor(z, 512, basis=Q)
        torch.testing.assert_close(future.reshape(5, 4, 4).norm(dim=-1),
                                   z.reshape(5, 4, 4).norm(dim=-1), rtol=2e-15, atol=2e-15)
        self.assertEqual(model.get_active_factor(torch.zeros(20, 16)), 0)

    def test_model_rollouts_cache_one_basis_and_ar_matches_direct_after_512_steps(self):
        model = M1_IJEPA1D(L=3, C=2, d=8, dtype=torch.float64)
        with torch.no_grad():
            model.predictor.S.normal_(std=0.08)
        z0 = torch.randn(3, 8, dtype=torch.float64)
        with mock.patch("torch.matrix_exp", wraps=torch.matrix_exp) as matrix_exp:
            ar = model.rollout_autoregressive(z0, 512)
        self.assertEqual(matrix_exp.call_count, 1)
        with mock.patch("torch.matrix_exp", wraps=torch.matrix_exp) as matrix_exp:
            direct = model.rollout_direct(z0, 512)
        self.assertEqual(matrix_exp.call_count, 1)
        torch.testing.assert_close(ar, direct, rtol=2e-10, atol=2e-10)
        torch.testing.assert_close(ar.norm(dim=-1), z0.norm(dim=-1)[:, None].expand(-1, 512),
                                   rtol=3e-12, atol=3e-12)
        self.assertEqual(model.rollout_direct(z0, 0).shape, (3, 0, 8))

    def test_m3_normalizes_encoders_but_does_not_recenter_rotated_states(self):
        model = M3_IJEPA_SIGReg(L=3, C=2, d=8, normalize_embeddings=True, dtype=torch.float64)
        self.assertIsInstance(model.predictor, ActionAngleKoopmanPredictor)
        self.assertFalse(any(p.requires_grad for p in model.target_encoder.parameters()))
        with torch.no_grad():
            model.predictor.S.normal_(std=0.1)
        x = torch.randn(4, 3, 2, dtype=torch.float64)
        z = model.encoder(x)
        torch.testing.assert_close(z.mean(dim=-1), torch.zeros(4, dtype=z.dtype), rtol=0, atol=2e-16)
        torch.testing.assert_close(z.square().mean(dim=-1), torch.ones(4, dtype=z.dtype), rtol=0, atol=8e-16)
        torch.testing.assert_close(model.encoder(-x), -z, rtol=0, atol=0)
        torch.testing.assert_close(model.target_encoder(x), z, rtol=0, atol=0)
        direct = model.rollout_direct(z, 512)
        ar = model.rollout_autoregressive(z, 512)
        torch.testing.assert_close(ar, direct, rtol=3e-10, atol=3e-10)
        torch.testing.assert_close(ar.square().mean(dim=-1), torch.ones(4, 512, dtype=z.dtype),
                                   rtol=0, atol=3e-12)
        self.assertGreater(direct[:, -1].mean(dim=-1).abs().max().item(), 1e-4)

    def test_random_baseline_is_frozen_odd_encoder_and_unstructured_predictor(self):
        model = M0_RandomJEPA(L=3, C=2, d=8, H_max=512, arch_type="mlp", dtype=torch.float64)
        self.assertIsInstance(model.encoder.network, OddResidualMLP)
        self.assertIsInstance(model.predictor, HorizonPredictor)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        x = torch.randn(4, 3, 2, dtype=torch.float64)
        torch.testing.assert_close(model.encoder(-x), -model.encoder(x), rtol=0, atol=0)
        self.assertEqual(model.rollout_direct(model.encoder(x), 512).shape, (4, 512, 8))

    def test_explicit_legacy_configuration_retains_archive_modules_and_keys(self):
        model = M2_JEPA_OPF(L=3, C=2, d=8, K=4, mlp_variant="legacy_gelu",
                            predictor_type="legacy_mlp", dtype=torch.float64)
        self.assertIsInstance(model.encoder.network[1], nn.GELU)
        self.assertEqual(len(model.predictors), 4)
        self.assertIsInstance(model.predictors[0], HorizonPredictor)
        self.assertIn("context_encoder.network.0.bias", model.state_dict())
        self.assertIn("predictor.predictors.0.pos_emb", model.state_dict())
        m3 = M3_IJEPA_SIGReg(L=3, C=2, d=8, mlp_variant="legacy_gelu",
                             predictor_type="legacy_mlp", normalize_embeddings=True)
        self.assertTrue(hasattr(m3.predictor, "base_predictor"))

    def test_invalid_rotational_dimensions_horizons_and_basis_are_rejected(self):
        for kwargs in ({"d": 7}, {"d": 12, "K": 4}, {"dt_pred": 0},
                       {"dt_pred": float("nan")}, {"K": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActionAngleKoopmanPredictor(**kwargs)
        predictor = self._predictor()
        z = torch.randn(3, 8, dtype=torch.float64)
        for k in (0, -1, True, 1.5, torch.tensor([1, 2]), torch.tensor([1., 2., 3.]),
                  torch.tensor([1, 0, 3]), torch.ones(3, 1, dtype=torch.long)):
            with self.subTest(k=k), self.assertRaises(ValueError):
                predictor(z, k)
        with self.assertRaises(ValueError):
            predictor(z, 1, basis=torch.eye(4, dtype=torch.float64))
        with self.assertRaises(ValueError):
            SequenceEncoder(arch_type="mlp", mlp_variant="unknown")


if __name__ == "__main__":
    unittest.main()
