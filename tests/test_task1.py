"""Independent implementation tests; scientific acceptance is reported separately.

Run with ``python -m unittest tests.test_task1`` from the repository root.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    KeplerWindowDataset,
    build_dataset_circ,
    build_dataset_counterfactual,
    build_dataset_ellip,
)
from src.simulator import (
    compute_orbital_invariants,
    generate_kepler_analytical,
    integrate_verlet,
)
from src.spectral import (
    evaluate_kepler_regression,
    freq_s1_discrete_fft,
    freq_s2_interpolated_fft,
    freq_s3a_dmd_koopman,
    freq_s3b_phase_unwrap,
    project_to_2d_mode,
)


class SimulatorTests(unittest.TestCase):
    def test_circles_match_closed_form_positions_and_velocities(self):
        a = np.array([[0.5], [2.0]])
        t = np.linspace(-2.0, 12.0, 137)[None, :]
        initial_phase = np.array([[0.3], [1.2]])
        mu = 2.5
        states = generate_kepler_analytical(a, np.zeros_like(a), t, mu=mu, M0=initial_phase)
        phase = np.sqrt(mu / a**3) * t + initial_phase
        speed = np.sqrt(mu / a)
        expected = np.stack((a * np.cos(phase), a * np.sin(phase),
                             -speed * np.sin(phase), speed * np.cos(phase)), axis=-1)
        self.assertEqual(states.shape, (2, 137, 4))
        self.assertEqual(states.dtype, np.float64)
        assert_allclose(states, expected, atol=3e-14, rtol=3e-14)

    def test_elliptical_apsides_and_rotation(self):
        a, e, mu = 2.0, 0.4, 3.0
        phase = np.array([[0.0], [np.pi]])
        states = generate_kepler_analytical(np.full((2, 1), a), np.full((2, 1), e),
                                            np.array([[0.0]]), mu=mu, M0=phase)
        expected = np.array([[a * (1 - e), 0, 0, np.sqrt(mu / a * (1 + e) / (1 - e))],
                             [-a * (1 + e), 0, 0, -np.sqrt(mu / a * (1 - e) / (1 + e))]])
        assert_allclose(states[:, 0], expected, atol=2e-15, rtol=2e-15)
        angle = 0.73
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        rotated = generate_kepler_analytical(np.full((2, 1), a), np.full((2, 1), e),
                                             np.array([[0.0]]), mu=mu, M0=phase,
                                             omega_arg=angle)
        assert_allclose(rotated[..., :2], states[..., :2] @ rotation.T, atol=2e-15)
        assert_allclose(rotated[..., 2:], states[..., 2:] @ rotation.T, atol=2e-15)

    def test_analytical_velocity_is_position_derivative(self):
        a, e, t = np.array([[0.8], [2.3]]), np.array([[0.55], [0.2]]), np.array([[0.4, 1.7]])
        h = 1e-6
        states = generate_kepler_analytical(a, e, t, omega_arg=1.1, M0=0.8)
        before = generate_kepler_analytical(a, e, t - h, omega_arg=1.1, M0=0.8)
        after = generate_kepler_analytical(a, e, t + h, omega_arg=1.1, M0=0.8)
        assert_allclose((after[..., :2] - before[..., :2]) / (2 * h),
                        states[..., 2:], rtol=3e-8, atol=2e-9)

    def test_near_parabolic_kepler_equation_residual(self):
        e = np.array([[0.9999]])
        mean = np.array([[-2.0, -0.01, -1e-6, 0.0, 1e-6, 0.01, 2.0]])
        states = generate_kepler_analytical(np.ones((1, 1)), e, mean)
        eccentric_anomaly = np.arctan2(states[..., 1] / np.sqrt(1 - e**2),
                                       states[..., 0] + e)
        assert_allclose(eccentric_anomaly - e * np.sin(eccentric_anomaly), mean,
                        atol=2e-14, rtol=2e-13)

    def test_analytical_invariants_match_orbit_parameters(self):
        a, e, mu = np.array([[0.5], [1.7], [5.0]]), np.array([[0.6], [0.2], [0.0]]), 1.3
        states = generate_kepler_analytical(a, e, np.linspace(0, 20, 1001)[None, :],
                                            mu=mu, omega_arg=0.6, M0=0.2)
        invariants = compute_orbital_invariants(states, mu=mu)
        assert_allclose(invariants["energy"], np.broadcast_to(-mu / (2 * a), (3, 1001)),
                        rtol=2e-14, atol=2e-14)
        assert_allclose(invariants["angular_momentum"],
                        np.broadcast_to(np.sqrt(mu * a * (1 - e**2)), (3, 1001)),
                        rtol=2e-14, atol=2e-14)
        assert_allclose(invariants["a_eff"], np.broadcast_to(a, (3, 1001)), rtol=3e-14)
        self.assertLess(invariants["energy_drift"].max(), 3e-14)
        self.assertLess(invariants["angular_momentum_drift"].max(), 3e-14)

    def test_verlet_includes_initial_state_and_is_second_order(self):
        r0, v0 = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
        errors = []
        for dt in (0.04, 0.02, 0.01):
            steps = round(2.0 / dt) + 1
            states = integrate_verlet(r0, v0, steps, dt=dt)
            self.assertEqual(states.dtype, np.float64)
            assert_array_equal(states[:, 0, :2], r0)
            assert_array_equal(states[:, 0, 2:], v0)
            expected = np.array([np.cos(2.0), np.sin(2.0), -np.sin(2.0), np.cos(2.0)])
            errors.append(np.linalg.norm(states[0, -1] - expected))
            self.assertLess(compute_orbital_invariants(states)["angular_momentum_drift"].max(),
                            2e-14)
        assert_allclose(np.array(errors[:-1]) / errors[1:], 4.0, rtol=0.01)
        assert_array_equal(r0, [[1.0, 0.0]])
        assert_array_equal(v0, [[0.0, 1.0]])
        assert_array_equal(integrate_verlet(r0, v0, 1), [[[1.0, 0.0, 0.0, 1.0]]])

    def test_zero_invariant_has_undefined_relative_drift(self):
        # Binary-exact parabolic state: v^2/2 = mu/r = 2, and L_z = 0.
        invariants = compute_orbital_invariants(np.array([[[1.0, 0.0, 2.0, 0.0]]]), mu=2.0)
        self.assertTrue(np.isnan(invariants["energy_drift"][0]))
        self.assertTrue(np.isnan(invariants["angular_momentum_drift"][0]))
        self.assertTrue(np.isinf(invariants["a_eff"][0, 0]))

    def test_simulator_rejects_invalid_inputs(self):
        a, e, t = np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 2))
        cases = [
            lambda: generate_kepler_analytical(a.ravel(), e, t),
            lambda: generate_kepler_analytical(-a, e, t),
            lambda: generate_kepler_analytical(a, a, t),
            lambda: generate_kepler_analytical(a, -a, t),
            lambda: generate_kepler_analytical(a, e, t.ravel()),
            lambda: generate_kepler_analytical(a, e, t, omega_arg=np.zeros(1)),
            lambda: generate_kepler_analytical(a, e, t, mu=0),
            lambda: generate_kepler_analytical(a.astype(complex), e, t),
            lambda: integrate_verlet(np.zeros((1, 2)), np.zeros((1, 2)), 3),
            lambda: integrate_verlet(np.ones((1, 2)), np.ones((2, 2)), 3),
            lambda: integrate_verlet(np.ones((1, 2)), np.ones((1, 2)), 0),
            lambda: integrate_verlet(np.ones((1, 2)), np.ones((1, 2)), True),
            lambda: integrate_verlet(np.ones((1, 2)), np.ones((1, 2)), 3, dt=np.nan),
            lambda: compute_orbital_invariants(np.zeros((1, 2, 4))),
            lambda: compute_orbital_invariants(np.ones((1, 4))),
            lambda: compute_orbital_invariants(np.full((1, 2, 4), np.inf)),
        ]
        for case in cases:
            with self.subTest(case=cases.index(case)), self.assertRaises(ValueError):
                case()


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circular = {split: build_dataset_circ(split) for split in ("train", "test_id", "test_ood")}
        cls.elliptical = build_dataset_ellip("test_id")
        cls.counterfactual = build_dataset_counterfactual()

    def test_circular_grid_shapes_and_disjoint_orbits(self):
        for split, count in (("train", 200), ("test_id", 50), ("test_ood", 50)):
            batch = self.circular[split]
            self.assertEqual(batch["trajectories"].shape, (count, 1024, 4))
            self.assertEqual(batch["trajectories"].dtype, np.float64)
            self.assertEqual(batch["split"], split)
            self.assertEqual(batch["dt"], 0.01)
            assert_array_equal(batch["meta"]["e"], np.zeros(count))
            assert_allclose(np.linalg.norm(batch["trajectories"][..., :2], axis=-1),
                            np.broadcast_to(batch["meta"]["a"][:, None], (count, 1024)),
                            rtol=2e-15, atol=2e-15)
        train = self.circular["train"]["meta"]["a"]
        test = self.circular["test_id"]["meta"]["a"]
        self.assertFalse(np.isclose(train[:, None], test, rtol=1e-12, atol=0).any())
        self.assertGreater(test.min(), 0.5)
        self.assertLess(test.max(), 5.0)
        ood = self.circular["test_ood"]["meta"]["a"]
        self.assertTrue(np.all((ood[:25] >= 0.25) & (ood[:25] <= 0.45)))
        self.assertTrue(np.all((ood[25:] >= 5.5) & (ood[25:] <= 8.0)))

    def test_builders_are_deterministic_and_seeds_affect_phases(self):
        for factory, existing in ((lambda: build_dataset_circ("test_id"), self.circular["test_id"]),
                                  (lambda: build_dataset_counterfactual(), self.counterfactual)):
            rebuilt = factory()
            assert_array_equal(rebuilt["trajectories"], existing["trajectories"])
            for key in existing["meta"]:
                assert_array_equal(rebuilt["meta"][key], existing["meta"][key])
        rebuilt_ellipses = build_dataset_ellip("test_id")
        for variant in self.elliptical:
            assert_array_equal(rebuilt_ellipses[variant]["trajectories"],
                               self.elliptical[variant]["trajectories"])
        changed = build_dataset_circ("test_id", seed=43)
        assert_array_equal(changed["meta"]["a"], self.circular["test_id"]["meta"]["a"])
        self.assertFalse(np.array_equal(changed["meta"]["M0"],
                                        self.circular["test_id"]["meta"]["M0"]))

    def test_even_context_is_centered_at_continuous_apsis(self):
        peri, apo = self.elliptical["perihelion_init"], self.elliptical["aphelion_init"]
        for key in ("a", "e", "omega_arg"):
            assert_array_equal(peri["meta"][key], apo["meta"][key])
        for batch, mean_apsis, sign in ((peri, 0.0, 1), (apo, np.pi, -1)):
            meta = batch["meta"]
            a, e, angle = meta["a"], meta["e"], meta["omega_arg"]
            self.assertTrue(np.all((e >= 0.2) & (e <= 0.6)))
            assert_allclose(meta["M0"] + np.sqrt(1 / a**3) * 0.075, mean_apsis,
                            atol=5e-16, rtol=0)
            center = generate_kepler_analytical(a[:, None], e[:, None], np.array([[0.075]]),
                                                M0=meta["M0"][:, None], omega_arg=angle[:, None])
            radius = a * (1 - sign * e)
            expected = sign * radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
            assert_allclose(center[:, 0, :2], expected, atol=8e-15, rtol=2e-15)
            # No discrete sample is the center: indices 7 and 8 bracket it symmetrically.
            radii = np.linalg.norm(batch["trajectories"][:, 7:9, :2], axis=-1)
            assert_allclose(radii[:, 0], radii[:, 1], atol=5e-15, rtol=2e-15)
            self.assertTrue(np.all(sign * (radii[:, 0] - radius) > 0))

    def test_counterfactual_metadata_follows_initial_energy_and_momentum(self):
        batch, meta = self.counterfactual, self.counterfactual["meta"]
        states = batch["trajectories"]
        self.assertEqual(states.shape, (50, 2048, 4))
        assert_array_equal(states[:, 0, 0], meta["initial_radius"])
        assert_array_equal(states[:, 0, [1, 2]], np.zeros((50, 2)))
        self.assertTrue(np.all((meta["alpha"] >= 0.75) & (meta["alpha"] <= 1.15)))
        self.assertTrue(np.all(meta["alpha"] != 1))
        invariants = compute_orbital_invariants(states[:, :1])
        assert_allclose(meta["a_true"], invariants["a_eff"][:, 0], rtol=2e-15)
        assert_allclose(meta["omega_init"],
                        invariants["angular_momentum"][:, 0] / meta["initial_radius"]**2,
                        rtol=2e-15)
        assert_allclose(meta["omega_true"], meta["a_true"]**-1.5, rtol=2e-15)
        recovered_e = np.sqrt(1 + 2 * invariants["energy"][:, 0]
                              * invariants["angular_momentum"][:, 0]**2)
        assert_allclose(meta["e"], recovered_e, atol=2e-13, rtol=2e-13)
        self.assertTrue(np.all(meta["a_true"] != meta["initial_radius"]))

    def test_builder_invalid_split_and_seed(self):
        for factory in (build_dataset_circ, build_dataset_ellip):
            with self.assertRaises(ValueError):
                factory("validation")
            for seed in (-1, True, 0.5):
                with self.subTest(factory=factory.__name__, seed=seed), self.assertRaises(ValueError):
                    factory("train", seed=seed)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.states = np.arange(2 * 9 * 4, dtype=np.float64).reshape(2, 9, 4)
        self.dataset = KeplerWindowDataset(self.states, L=3, H=2, meta={"a": np.array([1.0, 2.0])})

    def test_all_windows_include_final_target_and_do_not_cross_orbits(self):
        self.assertEqual(len(self.dataset), 10)
        for index in range(len(self.dataset)):
            orbit, start = divmod(index, 5)
            context, targets, meta = self.dataset[index]
            assert_array_equal(context.numpy(), self.states[orbit, start:start + 3])
            for horizon in range(1, 3):
                assert_array_equal(targets[horizon - 1].numpy(),
                                   self.states[orbit, start + horizon:start + horizon + 3])
            self.assertEqual((meta["orbit_index"], meta["start_index"]), (orbit, start))
            self.assertEqual(context.dtype, torch.float32)
            self.assertEqual(targets.dtype, torch.float32)
        assert_array_equal(self.dataset[-1][1][-1].numpy(), self.states[-1, -3:])
        two_channel = KeplerWindowDataset(self.states, L=3, H=2, channels=2)
        self.assertEqual(two_channel[0][0].shape, (3, 2))
        self.assertEqual(two_channel[0][1].shape, (2, 3, 2))

    def test_returned_tensors_own_memory_and_metadata_is_copied(self):
        before = self.states.copy()
        context, targets, meta = self.dataset[0]
        context.fill_(-10)
        targets.fill_(-20)
        meta["a"].fill_(-30)
        assert_array_equal(self.states, before)
        assert_array_equal(self.dataset[0][0].numpy(), before[0, :3])
        self.assertEqual(self.dataset[0][2]["a"].item(), 1.0)
        source_meta = np.array([4.0, 5.0])
        dataset = KeplerWindowDataset(self.states, L=3, H=2, meta={"a": source_meta})
        source_meta[:] = -1
        self.assertEqual(dataset[0][2]["a"].item(), 4.0)

    def test_default_dataloader_collates_windows_and_metadata(self):
        context, targets, meta = next(iter(DataLoader(self.dataset, batch_size=3, num_workers=0)))
        self.assertEqual(context.shape, (3, 3, 4))
        self.assertEqual(targets.shape, (3, 2, 3, 4))
        self.assertEqual(meta["a"].dtype, torch.float32)
        self.assertEqual(meta["dt"].dtype, torch.float32)
        assert_array_equal(meta["orbit_index"].numpy(), [0, 0, 0])
        assert_array_equal(meta["start_index"].numpy(), [0, 1, 2])

    def test_window_rejects_invalid_inputs_and_indices(self):
        cases = [
            lambda: KeplerWindowDataset(np.ones((2, 4))),
            lambda: KeplerWindowDataset(self.states[:, :4], L=3, H=2),
            lambda: KeplerWindowDataset(self.states, L=True, H=2),
            lambda: KeplerWindowDataset(self.states, L=3, H=1.5),
            lambda: KeplerWindowDataset(self.states, L=3, H=2, channels=3),
            lambda: KeplerWindowDataset(self.states[..., :2], L=3, H=2, channels=4),
            lambda: KeplerWindowDataset(self.states.astype(complex), L=3, H=2),
            lambda: KeplerWindowDataset(np.full_like(self.states, np.nan), L=3, H=2),
            lambda: KeplerWindowDataset(self.states, L=3, H=2, meta={"a": [1.0]}),
            lambda: KeplerWindowDataset(self.states, L=3, H=2, meta={"a": [1.0, np.inf]}),
        ]
        for case in cases:
            with self.subTest(case=cases.index(case)), self.assertRaises(ValueError):
                case()
        for index in (len(self.dataset), -len(self.dataset) - 1):
            with self.assertRaises(IndexError):
                self.dataset[index]
        for index in (True, 1.5, "0"):
            with self.assertRaises(TypeError):
                self.dataset[index]


class SpectralTests(unittest.TestCase):
    estimators = (freq_s1_discrete_fft, freq_s2_interpolated_fft,
                  freq_s3a_dmd_koopman, freq_s3b_phase_unwrap)

    @staticmethod
    def circle(omega, count=256, dt=0.01):
        phase = omega * np.arange(count) * dt + 0.4
        return np.column_stack((np.cos(phase), np.sin(phase)))

    def test_projection_preserves_2d_axes_and_rank_two_gram_matrix(self):
        values = self.circle(0.7)
        before = values.copy()
        centered = values - values.mean(axis=0)
        assert_array_equal(project_to_2d_mode(values), centered)
        embedding = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]) / np.sqrt(2)
        projected = project_to_2d_mode(values @ embedding + np.arange(4))
        self.assertEqual(projected.dtype, np.float64)
        self.assertEqual(projected.shape, (256, 2))
        assert_allclose(projected @ projected.T, centered @ centered.T, atol=1e-15, rtol=2e-13)
        assert_allclose(projected.mean(axis=0), 0, atol=1e-15)
        assert_array_equal(values, before)

    def test_raw_short_circle_recovers_frequency_but_centering_biases_it(self):
        omega, dt = 0.7, 0.01
        raw = self.circle(omega, dt=dt)
        before = raw.copy()
        centered = project_to_2d_mode(raw)
        for estimator in (freq_s3a_dmd_koopman, freq_s3b_phase_unwrap):
            self.assertAlmostEqual(estimator(raw, dt), omega, delta=1e-12)
            # Temporal centering of a fraction of a period changes the origin.
            self.assertGreater(abs(estimator(centered, dt) - omega) / omega, 0.01)
        assert_array_equal(raw, before)

    def test_discrete_fft_returns_known_bin_and_ignores_translation(self):
        count, dt, bin_index = 512, 0.025, 9
        omega = 2 * np.pi * bin_index / (count * dt)
        values = self.circle(omega, count, dt)
        self.assertAlmostEqual(freq_s1_discrete_fft(values, dt), omega, delta=1e-14)
        self.assertAlmostEqual(freq_s1_discrete_fft(values + [20.0, -12.0], dt), omega, delta=1e-14)

    def test_hann_interpolation_refines_an_off_bin_peak(self):
        count, dt = 2048, 0.01
        bin_width = 2 * np.pi / (count * dt)
        omega = 12.25 * bin_width
        values = self.circle(omega, count, dt)
        discrete_error = abs(freq_s1_discrete_fft(values, dt) - omega)
        refined_error = abs(freq_s2_interpolated_fft(values, dt) - omega)
        self.assertLess(refined_error, discrete_error)
        self.assertLess(refined_error, 0.02 * bin_width)
        alternating = np.column_stack(((-1.0)**np.arange(count), np.zeros(count)))
        self.assertAlmostEqual(freq_s2_interpolated_fft(alternating, dt), np.pi / dt, delta=1e-12)

    def test_phase_frequency_is_independent_of_rotation_direction(self):
        values = self.circle(2.3, count=600)
        reflected = values * [1.0, -1.0]
        backwards = values[::-1]
        for trajectory in (values, reflected, backwards):
            self.assertAlmostEqual(freq_s3b_phase_unwrap(trajectory), 2.3, delta=2e-14)
            self.assertAlmostEqual(freq_s3a_dmd_koopman(trajectory), 2.3, delta=2e-13)

    def test_higher_dimensional_estimators_use_the_declared_projection(self):
        values = self.circle(2 * np.pi * 5 / (512 * 0.01), count=512)
        embedded = np.column_stack((values, 2 * values[:, 0], -3 * values[:, 1]))
        projected = project_to_2d_mode(embedded)
        for estimator in self.estimators:
            with self.subTest(estimator=estimator.__name__):
                self.assertAlmostEqual(estimator(embedded), estimator(projected), delta=1e-12)

    def test_estimators_reject_invalid_and_degenerate_trajectories(self):
        for estimator in self.estimators:
            for values, dt in ((np.zeros((8, 2)), 0.01), (np.ones((8, 2)), 0.01),
                               (self.circle(1.0), 0), (self.circle(1.0), np.inf),
                               (np.ones((8, 1)), 0.01), (np.ones((1, 2)), 0.01),
                               (np.full((8, 2), np.nan), 0.01),
                               (np.ones((8, 2), dtype=complex), 0.01)):
                with self.subTest(estimator=estimator.__name__, shape=values.shape, dt=dt):
                    with self.assertRaises(ValueError):
                        estimator(values, dt)
        line = np.column_stack((np.arange(10.0), np.zeros(10)))
        with self.assertRaises(ValueError):
            freq_s3a_dmd_koopman(line)
        time = np.arange(10) * 0.01
        with self.assertRaises(ValueError):
            freq_s3a_dmd_koopman(np.column_stack((np.exp(time), np.exp(2 * time))))
        crossing_origin = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            freq_s3b_phase_unwrap(crossing_origin)

    def test_kepler_regression_and_residual_metrics(self):
        axes = np.geomspace(0.5, 8.0, 40)
        exact = evaluate_kepler_regression(axes, 1.7 * axes**-1.5)
        self.assertAlmostEqual(exact["slope"], -1.5, delta=1e-14)
        self.assertAlmostEqual(exact["intercept"], np.log(1.7), delta=1e-14)
        self.assertLess(exact["slope_abs_error"], 1e-14)
        self.assertLess(exact["one_minus_r2"], 1e-28)
        self.assertLess(exact["log10_one_minus_r2"], -28)
        frequencies = axes**-1.3 * np.exp(0.1 * np.sin(np.arange(len(axes))))
        result = evaluate_kepler_regression(axes, frequencies)
        x, y = np.log(axes), np.log(frequencies)
        slope, intercept = np.polyfit(x, y, 1)
        ratio = np.sum((y - (slope * x + intercept))**2) / np.sum((y - y.mean())**2)
        self.assertAlmostEqual(result["slope"], slope, delta=2e-15)
        self.assertAlmostEqual(result["one_minus_r2"], ratio, delta=2e-16)
        self.assertAlmostEqual(result["r2"], 1 - ratio, delta=2e-16)
        self.assertAlmostEqual(result["log10_one_minus_r2"], np.log10(ratio), delta=2e-14)

    def test_regression_constant_frequency_and_invalid_inputs(self):
        result = evaluate_kepler_regression([1, 2, 3], [4, 4, 4])
        self.assertEqual(result["slope"], 0.0)
        for key in ("r2", "one_minus_r2", "log10_one_minus_r2"):
            self.assertTrue(np.isnan(result[key]))
        for axes, frequencies in (([1], [2]), ([1, 1], [2, 3]), ([0, 1], [1, 2]),
                                  ([1, 2], [-1, 2]), ([1, 2], [1]),
                                  ([[1, 2]], [[1, 2]]), ([1, np.inf], [1, 2]),
                                  ([1 + 0j, 2], [1, 2])):
            with self.subTest(axes=axes, frequencies=frequencies), self.assertRaises(ValueError):
                evaluate_kepler_regression(axes, frequencies)


if __name__ == "__main__":
    unittest.main()
