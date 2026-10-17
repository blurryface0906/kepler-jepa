"""Small deterministic tests of Task 3 scientific evaluation, never training."""

import unittest

import numpy as np
import torch
from torch import nn

from src.evaluation import (METHODS, evaluate_frequencies, evaluate_regimes,
                            evaluate_scale_preservation, regime_metadata)


def circles(radii, count=256):
    phase = np.arange(count, dtype=np.float64) * (2 * np.pi / count)
    return np.asarray(radii)[:, None, None] * np.column_stack((np.cos(phase), np.sin(phase)))[None]


class ScalePreservationTests(unittest.TestCase):
    def setUp(self):
        self.axes = np.geomspace(0.5, 5.0, 12)

    def test_scale_and_energy_perfect_linear_probes(self):
        states = circles(self.axes)
        states[:, :, 0] += 3 * self.axes[:, None]
        result = evaluate_scale_preservation(states, self.axes)
        self.assertAlmostEqual(result["scale_r2_cycle_radius"], 1.0, places=14)
        self.assertAlmostEqual(result["scale_r2_state_norm"], 1.0, places=14)
        self.assertAlmostEqual(result["energy_r2"], 1.0, places=14)
        self.assertEqual(result["cycle_radius_direction"], "increasing")
        self.assertAlmostEqual(result["cycle_radius_spearman"], 1.0, places=14)

    def test_s_and_r_match_prescribed_formulas(self):
        rng = np.random.default_rng(88)
        states = rng.normal(size=(12, 19, 7))
        result = evaluate_scale_preservation(states, self.axes, mu=2.5)
        centered = states - states.mean(axis=1, keepdims=True)
        expected_s = np.log(np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1) + 1e-12))
        expected_r = np.log(np.sqrt(np.mean(np.sum(states**2, axis=2), axis=1) + 1e-12))
        np.testing.assert_allclose(result["s"], expected_s, atol=1e-15)
        np.testing.assert_allclose(result["r"], expected_r, atol=1e-15)
        np.testing.assert_allclose(result["log_abs_energy"], np.log(2.5 / (2 * self.axes)), atol=1e-15)

    def test_zero_latent_has_zero_r2_not_perfect_noise_fit(self):
        result = evaluate_scale_preservation(np.zeros((12, 10, 4)), self.axes)
        for key in ("scale_r2_cycle_radius", "scale_r2_state_norm", "energy_r2"):
            self.assertEqual(result[key], 0.0)
        self.assertEqual(result["fits"]["energy"]["active_features"], [])
        self.assertTrue(np.isnan(result["cycle_radius_spearman"]))

    def test_tiny_apparent_radius_information_is_disclosed_as_constant(self):
        states = circles(1 + self.axes * 1e-9)
        result = evaluate_scale_preservation(states, self.axes)
        self.assertEqual(result["scale_r2_cycle_radius"], 0.0)
        self.assertEqual(result["scale_r2_state_norm"], 0.0)
        self.assertEqual(result["fits"]["cycle_radius"]["dropped_constant_features"], ["s"])
        self.assertGreater(result["fits"]["cycle_radius"]["feature_std"]["s"], 0)
        precise = evaluate_scale_preservation(states, self.axes, numerical_feature_tol=1e-12)
        self.assertGreater(precise["scale_r2_cycle_radius"], 0.9)

    def test_constant_axes_response_is_undefined(self):
        result = evaluate_scale_preservation(circles(self.axes), np.ones(12))
        for key in ("scale_r2_cycle_radius", "scale_r2_state_norm", "energy_r2"):
            self.assertTrue(np.isnan(result[key]))
        self.assertEqual(result["fits"]["cycle_radius"]["status"], "undefined_constant_response")

    def test_cycle_radius_translation_invariance(self):
        states = circles(self.axes)
        translated = states + np.arange(12)[:, None, None] * np.array([2.0, -3.0])
        base = evaluate_scale_preservation(states, self.axes)
        shifted = evaluate_scale_preservation(translated, self.axes)
        np.testing.assert_allclose(base["s"], shifted["s"], atol=1e-14)
        self.assertFalse(np.allclose(base["r"], shifted["r"]))

    def test_unit_sphere_state_norm_does_not_fix_cycle_radius_on_short_arcs(self):
        phase = self.axes[:, None] * np.arange(64)[None] * 0.01
        states = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        result = evaluate_scale_preservation(states, self.axes)
        self.assertEqual(result["scale_r2_state_norm"], 0.0)
        self.assertGreater(result["scale_r2_cycle_radius"], 0.9)
        self.assertGreater(result["energy_r2"], 0.9)

    def test_decreasing_scale_direction_is_reported(self):
        result = evaluate_scale_preservation(circles(self.axes**-1), self.axes)
        self.assertAlmostEqual(result["scale_r2_cycle_radius"], 1.0, places=13)
        self.assertAlmostEqual(result["cycle_radius_spearman"], -1.0, places=13)
        self.assertEqual(result["cycle_radius_direction"], "decreasing")

    def test_nonfinite_and_invalid_inputs_are_rejected(self):
        values = circles(self.axes)
        bad = values.copy()
        bad[1, 0, 0] = np.inf
        for states, axes in ((bad, self.axes), (values, -self.axes),
                             (values, self.axes[:-1]), (values[0], self.axes)):
            with self.subTest(shape=states.shape):
                with self.assertRaises(ValueError):
                    evaluate_scale_preservation(states, axes)
        for mu in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                evaluate_scale_preservation(values, self.axes, mu=mu)


class _ToyEncoder(nn.Module):
    L, C, d = 3, 2, 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))

    def forward(self, x):
        return x[:, 0] * self.scale


class _ToyPredictor(nn.Module):
    H_max = 4

    def __init__(self):
        super().__init__()
        self.tokens = []

    def forward(self, z, k=1):
        if not 1 <= k <= self.H_max:
            raise ValueError("untrained token")
        self.tokens.append(k)
        return z + z.new_tensor([2 * k, 200 * k])


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = _ToyEncoder()
        self.predictor = _ToyPredictor()

    def encode_sequence(self, x, batch_size=None):
        shape = x.shape[:-2]
        return self.context_encoder(x.reshape(-1, 3, 2)).reshape(*shape, 2)


class RegimeTests(unittest.TestCase):
    def setUp(self):
        times = np.arange(30, dtype=np.float64)
        self.trajectories = np.tile(np.column_stack((times, 100 * times))[None], (2, 1, 1))
        self.model = _ToyModel()

    def test_all_regimes_have_aligned_future_windows_and_t4_is_short(self):
        result = evaluate_regimes(self.model, self.trajectories, N_eval=8, L=3,
                                  delta=2, H_train=4, batch_size=5)
        expected = self.trajectories[:, np.arange(1, 9) * 2]
        for name in ("T1", "T2", "T3"):
            np.testing.assert_allclose(result[name], expected)
        np.testing.assert_allclose(result["T4"], expected[:, :4])
        self.assertEqual(result["T4"].shape, (2, 4, 2))
        self.assertEqual(self.model.predictor.tokens[-4:], [1, 2, 3, 4])
        self.assertTrue(self.model.training)

    def test_t2_uses_truth_but_t3_and_t4_use_only_initial_window(self):
        changed = self.trajectories.copy()
        changed[:, 2:] += 70
        kwargs = dict(N_eval=8, L=3, delta=2, H_train=4)
        before = evaluate_regimes(self.model, self.trajectories, **kwargs)
        after = evaluate_regimes(self.model, changed, **kwargs)
        self.assertFalse(np.array_equal(before["T1"], after["T1"]))
        self.assertFalse(np.array_equal(before["T2"], after["T2"]))
        np.testing.assert_array_equal(before["T3"], after["T3"])
        np.testing.assert_array_equal(before["T4"], after["T4"])

    def test_start_offset_and_metadata_are_exact(self):
        metadata = regime_metadata(N_eval=8, L=3, delta=2, H_train=4, dt=0.01, start=1)
        self.assertEqual(metadata["macro_dt"], 0.02)
        self.assertEqual(metadata["required_raw_samples"], 20)
        self.assertEqual(metadata["window_start_indices"]["T4"], [3, 5, 7, 9])
        np.testing.assert_allclose(metadata["sample_times"]["T4"], [0.03, 0.05, 0.07, 0.09])
        result = evaluate_regimes(self.model, self.trajectories, N_eval=8, L=3,
                                  delta=2, H_train=4, start=1)
        np.testing.assert_array_equal(result["T1"], self.trajectories[:, np.arange(1, 9) * 2 + 1])

    def test_evaluation_restores_eval_mode_and_does_not_change_weights(self):
        self.model.eval()
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        evaluate_regimes(self.model, self.trajectories, N_eval=8, L=3, delta=2, H_train=4)
        self.assertFalse(self.model.training)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_enough_true_samples_are_required_without_padding(self):
        with self.assertRaises(ValueError):
            evaluate_regimes(self.model, self.trajectories[:, :18], N_eval=8, L=3, delta=2, H_train=4)

    def test_untrained_direct_token_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            evaluate_regimes(self.model, self.trajectories, N_eval=8, L=3, delta=2, H_train=5)
        self.assertEqual(self.model.predictor.tokens, [])

    def test_dtype_conversion_of_trained_weights_is_disallowed(self):
        with self.assertRaises(ValueError):
            evaluate_regimes(self.model, self.trajectories, N_eval=8, L=3,
                             delta=2, H_train=4, dtype=torch.float32)
        self.assertEqual(self.model.context_encoder.scale.dtype, torch.float64)

    def test_production_regime_lengths_and_macro_cadence(self):
        metadata = regime_metadata()
        self.assertEqual(metadata["macro_dt"], 0.04)
        self.assertEqual(metadata["required_raw_samples"], 1040)
        self.assertEqual(len(metadata["window_start_indices"]["T3"]), 256)
        self.assertEqual(len(metadata["window_start_indices"]["T4"]), 64)
        self.assertEqual(metadata["window_start_indices"]["T1"][0], 4)


class FrequencyEvaluationTests(unittest.TestCase):
    def test_harmonic_dmd_diff_recovers_kepler_without_true_parameters(self):
        axes = np.geomspace(0.5, 2.0, 12)
        time = np.arange(256) * 0.04
        phase = axes[:, None]**-1.5 * time[None]
        harmonic = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        result = evaluate_frequencies(harmonic, axes, dt=0.04)
        self.assertEqual(set(result), set(METHODS))
        self.assertEqual(result["S3a_diff"]["invalid"], 0)
        np.testing.assert_allclose(result["S3a_diff"]["omega_est"], axes**-1.5, rtol=1e-10)
        self.assertLess(result["S3a_diff"]["slope_abs_error"], 1e-10)

    def test_failed_orbit_is_retained_and_makes_full_regression_undefined(self):
        axes = np.geomspace(0.5, 2.0, 5)
        harmonic = circles(axes)
        harmonic[2] = np.nan
        result = evaluate_frequencies(harmonic, axes)
        for record in result.values():
            self.assertEqual(len(record["omega_est"]), 5)
            self.assertEqual(record["invalid"], 1)
            self.assertEqual(record["invalid_orbit_indices"], [2])
            self.assertTrue(np.isnan(record["r2"]))
            self.assertIsNotNone(record["errors"][2])

    def test_collapsed_trajectory_cannot_receive_valid_frequency(self):
        result = evaluate_frequencies(np.zeros((3, 64, 4)), np.array([1, 2, 3]))
        for record in result.values():
            self.assertEqual(record["invalid"], 3)
            self.assertTrue(np.isnan(record["slope"]))


if __name__ == "__main__":
    unittest.main()
