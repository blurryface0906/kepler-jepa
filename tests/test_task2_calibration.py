"""Calibration invariants, independent of scientific acceptance statistics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.simulator import (compute_orbital_invariants, generate_kepler_analytical,
                           integrate_verlet)
from src.spectral import (freq_s3a_dmd_koopman, freq_s3b_phase_unwrap,
                          project_to_2d_mode)


class SubstepTests(unittest.TestCase):
    def test_saved_grid_matches_fine_integration_and_default_is_unchanged(self):
        r0 = np.array([[0.7, 0.0], [1.3, 0.0]])
        v0 = np.array([[0.0, 1.4], [0.0, 0.8]])
        assert_array_equal(integrate_verlet(r0, v0, 51),
                           integrate_verlet(r0, v0, 51, substeps=1))
        coarse = integrate_verlet(r0, v0, 51, substeps=5)
        fine = integrate_verlet(r0, v0, 251, dt=0.002)
        self.assertEqual(coarse.shape, (2, 51, 4))
        assert_array_equal(coarse, fine[:, ::5])

    def test_substeps_reduce_elliptic_energy_error_quadratically(self):
        initial = generate_kepler_analytical(np.array([[0.5]]), np.array([[0.58]]),
                                             np.array([[0.0]]))[:, 0]
        drift = []
        for substeps in (1, 5):
            states = integrate_verlet(initial[:, :2], initial[:, 2:], 1024,
                                      substeps=substeps)
            invariants = compute_orbital_invariants(states)
            drift.append(invariants['energy_drift'][0])
            self.assertLess(invariants['angular_momentum_drift'][0], 5e-14)
        self.assertGreater(drift[0] / drift[1], 24.0)
        self.assertLess(drift[0] / drift[1], 26.0)
        # This independent orbit starts exactly at pericenter; the scientific
        # dataset instead starts .075 time units before it. Compare integrator
        # convergence here, without importing the dataset's empirical bound.
        self.assertLess(drift[1], 2e-4)

    def test_invalid_substep_counts_are_rejected(self):
        for count in (0, -1, True, 1.5, float('inf'), '5'):
            with self.subTest(substeps=count), self.assertRaises(ValueError):
                integrate_verlet(np.array([[1., 0.]]), np.array([[0., 1.]]),
                                 2, substeps=count)


class DifferenceEstimatorTests(unittest.TestCase):
    methods = (freq_s3a_dmd_koopman, freq_s3b_phase_unwrap)

    @staticmethod
    def circle(omega=0.7, count=256):
        phase = 0.3 + omega * np.arange(count) * 0.01
        return np.column_stack((np.cos(phase), np.sin(phase)))

    def test_short_arc_translation_invariance_and_no_recentring(self):
        raw = self.circle()
        centered = project_to_2d_mode(raw)
        differences = np.diff(centered, axis=0)
        # The finite-arc velocity mean is decidedly nonzero; removing it
        # reintroduces the same finite-arc origin bias the feature addresses.
        self.assertGreater(np.linalg.norm(differences.mean(axis=0)), 0.005)
        for method in self.methods:
            with self.subTest(method=method.__name__):
                before = centered.copy()
                self.assertAlmostEqual(method(centered, use_differences=True),
                                       0.7, delta=2e-11)
                self.assertAlmostEqual(method(raw + [13., -7.], use_differences=True),
                                       0.7, delta=2e-10)
                self.assertAlmostEqual(method(centered, use_differences=True),
                                       method(differences), delta=2e-13)
                self.assertGreater(abs(method(project_to_2d_mode(differences)) - 0.7),
                                   0.01)
                assert_array_equal(centered, before)

    def test_dmd_is_affine_invariant_but_phase_can_be_anisotropically_biased(self):
        transformed = self.circle() @ np.array([[3.0, 0.2], [0.0, 0.4]]) + [2., -4.]
        self.assertAlmostEqual(freq_s3a_dmd_koopman(transformed, use_differences=True),
                               0.7, delta=2e-10)
        self.assertGreater(abs(freq_s3b_phase_unwrap(transformed, use_differences=True) - 0.7),
                           0.1)

    def test_high_dimensional_input_matches_explicit_projection(self):
        raw = self.circle(1.2)
        embedding = np.column_stack((raw, 2 * raw[:, 0], -raw[:, 1])) + [1., 2., 3., 4.]
        for method in self.methods:
            self.assertAlmostEqual(method(embedding, use_differences=True),
                                   method(project_to_2d_mode(embedding), use_differences=True),
                                   delta=1e-12)

    def test_differences_require_extra_sample_and_nonconstant_velocity(self):
        for method, count in ((freq_s3a_dmd_koopman, 3), (freq_s3b_phase_unwrap, 2)):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method(self.circle(count=count), use_differences=True)
        straight = np.column_stack((np.arange(10.), np.zeros(10)))
        for method in self.methods:
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method(straight, use_differences=True)
            with self.assertRaises(ValueError):
                method(self.circle(), use_differences='yes')


if __name__ == '__main__':
    unittest.main()
