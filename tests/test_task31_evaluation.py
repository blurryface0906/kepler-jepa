"""Task 3.1 alignment, physical controls and honest constant-norm diagnostics."""

import unittest

import numpy as np
import torch
from torch import nn

from src.evaluation import (evaluate_frequencies, evaluate_regimes,
                            evaluate_scale_preservation, regime_metadata)


def physical_circles(axes, *, count=512, dt=0.04, mu=1.0):
    axes = np.asarray(axes)
    phase = np.sqrt(mu / axes[:, None] ** 3) * (np.arange(count)[None] * dt)
    speed = np.sqrt(mu / axes[:, None])
    return np.stack((axes[:, None] * np.cos(phase), axes[:, None] * np.sin(phase),
                     -speed * np.sin(phase), speed * np.cos(phase)), axis=-1)


class Task31PhysicalScaleTests(unittest.TestCase):
    def setUp(self):
        self.axes = np.geomspace(0.5, 2.0, 30)

    def test_physical_state_norm_nonmonotonicity_is_measured_not_hardcoded(self):
        states = physical_circles(self.axes)
        result = evaluate_scale_preservation(states, self.axes)
        self.assertGreater(result["phase_norm_r2"], 1 - 1e-12)
        self.assertGreater(result["axis_probe_r2"], 0.999999)
        feature = 0.5 * np.log(self.axes ** 2 + 1 / self.axes)
        target = np.log(self.axes)
        residual = target - np.polyval(np.polyfit(feature, target, 1), feature)
        expected = 1 - residual @ residual / np.sum((target - target.mean()) ** 2)
        self.assertAlmostEqual(result["physical_state_norm_axis_r2"], expected, places=13)
        self.assertAlmostEqual(result["state_norm_r2"], expected, places=13)
        self.assertLess(result["state_norm_r2"], 0.9)
        self.assertEqual(result["state_norm_r2"], result["scale_r2_state_norm"])
        # The actual mean over four channels has factor1/4; omitted constants
        # only shift the phase regression intercept.
        np.testing.assert_allclose(result["r_rms"],
                                   0.5 * np.log((self.axes ** 2 + 1 / self.axes) / 4),
                                   atol=3e-12)

    def test_constant_sphere_norm_zero_score_does_not_erase_frequency_information(self):
        states = physical_circles(self.axes)
        states /= np.linalg.norm(states, axis=-1, keepdims=True)
        result = evaluate_scale_preservation(states, self.axes)
        self.assertEqual(result["state_norm_r2"], 0.0)
        self.assertEqual(result["phase_norm_r2"], 0.0)
        self.assertTrue(result["state_norm_numerically_constant"])
        phase_fit = result["fits"]["phase_norm"]
        self.assertTrue(phase_fit["constant_response_convention_applied"])
        self.assertIn("zero_score_convention", phase_fit["status"])
        self.assertIn("conventional_r2", phase_fit)
        # Velocity can still carry the rate a^-1.5 even though state norm is
        # constant. A zero state-norm score is not proof of no radius information.
        self.assertGreater(result["axis_probe_r2"], 0.999999)
        self.assertEqual(result["fits"]["axis_probe"]["active_features"], ["v_rms"])

    def test_velocity_probe_uses_reported_macro_cadence(self):
        states = physical_circles(self.axes)
        result = evaluate_scale_preservation(states, self.axes, dt_pred=0.04)
        expected_v = np.log(np.sqrt(np.mean((np.diff(states, axis=1) / 0.04) ** 2,
                                            axis=(1, 2)) + 1e-12))
        np.testing.assert_allclose(result["v_rms"], expected_v, atol=1e-15)
        slower_units = evaluate_scale_preservation(states, self.axes, dt_pred=0.08)
        np.testing.assert_allclose(np.asarray(result["v_rms"]) - slower_units["v_rms"],
                                   np.log(2), atol=5e-11)
        with self.assertRaises(ValueError):
            evaluate_scale_preservation(states, self.axes, dt_pred=0)

    def test_constant_response_threshold_is_explicit_and_configurable(self):
        phase = np.arange(512)[None] * 0.03
        perturbation = (1 + 1e-8 * self.axes[:, None])
        states = perturbation[..., None] * np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        coarse = evaluate_scale_preservation(states, self.axes)
        fine = evaluate_scale_preservation(states, self.axes, numerical_feature_tol=1e-12)
        self.assertEqual(coarse["phase_norm_r2"], 0)
        self.assertTrue(coarse["state_norm_numerically_constant"])
        self.assertFalse(fine["state_norm_numerically_constant"])
        self.assertGreater(fine["phase_norm_r2"], 0.1)


class _StartEncoder(nn.Module):
    L, C, d = 16, 4, 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))

    def forward(self, windows):
        return windows[:, 0] * self.scale


class _CachedToyPredictor(nn.Module):
    H_max = 64
    supports_extrapolation = True

    def __init__(self):
        super().__init__()
        self.basis_calls = 0
        self.basis_seen = []

    def orthogonal_basis(self):
        self.basis_calls += 1
        return torch.eye(4, dtype=torch.float64)

    def forward(self, z, k=1, *, basis=None):
        self.basis_seen.append(basis)
        return z + 4 * k * z.new_tensor([1, 2, 3, 4])

    def predict_horizons(self, z, horizons, *, basis=None):
        self.basis_seen.append(basis)
        return z[:, None] + 4 * horizons[None, :, None] * z.new_tensor([1, 2, 3, 4])


class _AlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = _StartEncoder()
        self.predictor = _CachedToyPredictor()

    def encode_sequence(self, windows, batch_size=None):
        shape = windows.shape[:-2]
        return self.context_encoder(windows.reshape(-1, 16, 4)).reshape(*shape, 4)


class Task31RegimeTests(unittest.TestCase):
    def test_512_macrostates_and_extended_direct_horizons_remain_separate(self):
        states = np.tile((np.arange(2100)[:, None] * np.arange(1, 5)[None])[None], (2, 1, 1))
        model = _AlignedModel()
        result = evaluate_regimes(model, states, N_eval=512, H_extended=512)
        expected = states[:, np.arange(1, 513) * 4]
        self.assertEqual(set(result), {"T1", "T2", "T3", "T4", "T4_512"})
        for name in ("T1", "T2", "T3", "T4_512"):
            np.testing.assert_array_equal(result[name], expected)
        np.testing.assert_array_equal(result["T4"], expected[:, :64])
        self.assertEqual(model.predictor.basis_calls, 1)
        self.assertTrue(all(basis is model.predictor.basis_seen[0]
                            for basis in model.predictor.basis_seen))
        self.assertTrue(model.training)

    def test_span_and_origin_distance_are_distinct(self):
        metadata = regime_metadata(N_eval=512, H_extended=512)
        self.assertEqual(metadata["required_raw_samples"], 2064)
        self.assertEqual(metadata["window_start_indices"]["T1"][0], 4)
        self.assertEqual(metadata["window_start_indices"]["T1"][-1], 2048)
        self.assertEqual(metadata["sample_span"]["T1"], 20.44)
        self.assertEqual(metadata["last_horizon_from_origin"]["T1"], 20.48)
        self.assertEqual(metadata["sample_span"]["T4"], 2.52)
        self.assertEqual(metadata["extended_direct_regime"], "T4_512")

    def test_extrapolation_needs_explicit_predictor_support(self):
        states = np.zeros((2, 2100, 4))
        model = _AlignedModel()
        model.predictor.supports_extrapolation = False
        with self.assertRaises(ValueError):
            evaluate_regimes(model, states, N_eval=512, H_extended=512)
        self.assertEqual(model.predictor.basis_calls, 0)


class Task31FrequencyTests(unittest.TestCase):
    def test_uncentered_projection_retains_origin_on_partial_harmonic_arcs(self):
        axes = np.geomspace(0.5, 2, 12)
        time = np.arange(64) * 0.04
        phase = axes[:, None] ** -1.5 * time[None]
        harmonic = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        result = evaluate_frequencies(harmonic, axes, projection="uncentered_svd")
        np.testing.assert_allclose(result["S3b_std"]["omega_est"], axes ** -1.5,
                                   atol=2e-14)
        self.assertEqual(result["S3b_std"]["projection"], "uncentered_svd")

    def test_nonfinite_orbits_are_retained_with_optional_projection(self):
        axes = np.geomspace(0.5, 2, 12)
        states = physical_circles(axes)
        states[3] = np.nan
        result = evaluate_frequencies(states, axes, projection="uncentered_svd")
        for record in result.values():
            self.assertEqual(record["invalid_orbit_indices"], [3])
            self.assertEqual(len(record["omega_est"]), 12)
            self.assertTrue(np.isnan(record["r2"]))


if __name__ == "__main__":
    unittest.main()
