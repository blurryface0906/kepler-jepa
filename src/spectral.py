"""Frequency estimators and log-log Kepler regressions in float64.

``project_to_2d_mode`` explicitly subtracts the temporal mean.  A frequency
function accepts either its two-column output or a higher-dimensional latent
trajectory (which is projected automatically).  Two-column input to DMD and
phase unwrapping is used *unchanged*: the origin is part of those estimators.
The two FFT estimators subtract the temporal mean before spectral analysis.

This distinction matters for finite orbital arcs.  The sample mean of a short
arc is generally not the force center; centering such an arc can invalidate a
physical angular-frequency interpretation of DMD or atan2.  No estimator uses
orbital metadata, fits a circle, or corrects a latent phase using ground truth.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "project_to_2d_mode",
    "freq_s1_discrete_fft",
    "freq_s2_interpolated_fft",
    "freq_s3a_dmd_koopman",
    "freq_s3b_phase_unwrap",
    "evaluate_kepler_regression",
]


def _trajectory(values: np.ndarray, *, min_samples: int) -> np.ndarray:
    """Validate a finite real (time, features) array without modifying it."""
    if np.iscomplexobj(values):
        raise ValueError("Trajectory must be real-valued.")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Trajectory must contain real numeric values.") from error
    if result.ndim != 2 or result.shape[1] < 2:
        raise ValueError("Trajectory must have shape (T, d), with d >= 2.")
    if result.shape[0] < min_samples:
        raise ValueError(f"At least {min_samples} time samples are required.")
    if not np.all(np.isfinite(result)):
        raise ValueError("Trajectory must contain only finite values.")
    return result


def _mode(values: np.ndarray, dt: float, *, min_samples: int) -> tuple[np.ndarray, float]:
    """Validate sampling and preserve the origin of an explicit 2D input."""
    if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive scalar.")
    mode = _trajectory(values, min_samples=min_samples)
    if mode.shape[1] > 2:
        mode = project_to_2d_mode(mode)
    if not np.any(mode != mode[0]):
        raise ValueError("Frequency is undefined for a constant trajectory.")
    return mode, float(dt)


def project_to_2d_mode(Z: np.ndarray) -> np.ndarray:
    """Center a trajectory and return its leading two principal scores.

    For d == 2, only centering is performed: axis order, orientation and scale
    are preserved.  For d > 2 this is the unwhitened SVD/PCA score matrix,
    equivalent to ``PCA(n_components=2).fit_transform(Z)`` up to axis signs.
    Rank-deficient inputs remain rank-deficient; no missing mode is fabricated.
    """
    values = _trajectory(Z, min_samples=2)
    centered = values - values.mean(axis=0, keepdims=True)
    if values.shape[1] == 2:
        return centered
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(left[:, :2] * singular_values[:2], dtype=np.float64)


def _periodogram(U: np.ndarray, dt: float, *, hann: bool) -> tuple[np.ndarray, int, float]:
    mode, step = _mode(U, dt, min_samples=4 if hann else 3)
    count = mode.shape[0]
    centered = mode - mode.mean(axis=0, keepdims=True)
    if hann:
        centered = centered * np.hanning(count)[:, None]
    power = np.sum(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=1)
    if not np.all(np.isfinite(power)) or not np.any(power[1:] > 0):
        raise ValueError("Trajectory has no finite positive non-DC spectral power.")
    return power, count, step


def freq_s1_discrete_fft(U: np.ndarray, dt: float = 0.01) -> float:
    """Return the strongest non-DC rFFT bin in radians per time unit.

    Both coordinates contribute equally to total power.  Subtracting the mean
    is equivalent to suppressing the DC bin and improves numerical stability.
    Ties select the lowest-frequency bin, as with ``np.argmax``.
    """
    power, count, step = _periodogram(U, dt, hann=False)
    power[0] = 0.0
    peak = int(np.argmax(power))
    return float(2.0 * np.pi * peak / (count * step))


def freq_s2_interpolated_fft(U: np.ndarray, dt: float = 0.01) -> float:
    """Hann-windowed rFFT with three-bin log-power parabola interpolation.

    The temporal mean is removed before applying the Hann window.  DC power
    is set to zero before peak selection and interpolation as specified in
    Task 1. A last-bin peak or exactly flat parabola
    has no defined three-bin refinement and returns the discrete-bin estimate.
    Positive spectral powers are floored at float64's smallest normal number
    solely to make logarithms of zero-valued neighboring bins well-defined.
    """
    power, count, step = _periodogram(U, dt, hann=True)
    power[0] = 0.0
    peak = 1 + int(np.argmax(power[1:]))
    offset = 0.0
    if peak < power.size - 1:
        left, middle, right = np.log(
            np.maximum(power[peak - 1 : peak + 2], np.finfo(np.float64).tiny)
        )
        denominator = left - 2.0 * middle + right
        if denominator != 0.0:
            offset = float(0.5 * (left - right) / denominator)
    estimate = 2.0 * np.pi * (peak + offset) / (count * step)
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("The spectral peak does not admit a positive finite interpolation.")
    return float(estimate)


def _s3_mode(U: np.ndarray, dt: float, use_differences: bool,
             *, min_samples: int) -> tuple[np.ndarray, float]:
    if not isinstance(use_differences, (bool, np.bool_)):
        raise ValueError("use_differences must be a boolean.")
    mode, step = _mode(U, dt, min_samples=min_samples + int(use_differences))
    if use_differences:
        # Differencing annihilates translations. Do not remove the finite-arc
        # velocity mean: it is generally nonzero even for a harmonic orbit.
        mode, step = _mode(np.diff(mode, axis=0), step, min_samples=min_samples)
    return mode, step


def freq_s3a_dmd_koopman(U: np.ndarray, dt: float = 0.01,
                          use_differences: bool = False) -> float:
    """Fit the exact 2D least-squares transition and return its eigenfrequency.

    This implements the permitted two-state Exact DMD variant, without delay
    embedding: ``A = U[1:].T @ pinv(U[:-1].T)``.  The two-column input is used
    without centering, so an origin-centered circular orbit is an exact linear
    rotation even when less than one period is observed.  Mean-centered finite
    arcs and general elliptical coordinates need not obey that linear model.

    With use_differences=True, fit the transition of first differences without
    recentering. This removes arbitrary translations but does not linearize
    nonlinear encoders or elliptic Kepler dynamics.

    Rank-deficient predecessor states or an operator without a nonzero complex
    eigenvalue raise ValueError; neither case provides an oscillatory DMD mode.
    Sampling aliases frequencies above the Nyquist angular frequency pi/dt.
    """
    mode, step = _s3_mode(U, dt, use_differences, min_samples=3)
    previous = mode[:-1].T
    following = mode[1:].T
    if np.linalg.matrix_rank(previous) < 2:
        raise ValueError("DMD requires two independent predecessor-state directions.")
    operator = following @ np.linalg.pinv(previous)
    eigenvalues = np.linalg.eigvals(operator).astype(np.complex128)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("DMD produced non-finite eigenvalues.")
    oscillatory = eigenvalues[np.abs(eigenvalues.imag) > 0.0]
    if oscillatory.size == 0:
        raise ValueError("DMD transition has no oscillatory eigenmode.")
    eigenvalue = oscillatory[int(np.argmax(np.abs(oscillatory)))]
    return float(abs(np.angle(eigenvalue)) / step)


def freq_s3b_phase_unwrap(U: np.ndarray, dt: float = 0.01,
                         use_differences: bool = False) -> float:
    """Regress unwrapped atan2 phase on time and return the absolute slope.

    Explicit two-dimensional coordinates retain their supplied origin.  A
    circular orbit about that origin yields its mean motion; an ellipse yields
    a window-dependent average of true-anomaly rates.  PCA signs only change
    the direction of rotation, which the absolute slope removes.  General PCA
    scaling can deform phase and is not corrected here.  Sampling must resolve
    each angular increment to less than pi for unwrap to be unambiguous.

    With use_differences=True, regress the phase of first differences without
    recentering them. Translation invariance does not imply invariance to
    anisotropic scaling: an affine image of a circle can have nonuniform phase.
    """
    mode, step = _s3_mode(U, dt, use_differences, min_samples=2)
    if np.any(np.all(mode == 0.0, axis=1)):
        raise ValueError("Phase is undefined at samples located at the origin.")
    phase = np.unwrap(np.arctan2(mode[:, 1], mode[:, 0]))
    time = np.arange(mode.shape[0], dtype=np.float64) * step
    slope = np.polyfit(time, phase, 1)[0]
    return float(abs(slope))


def evaluate_kepler_regression(semi_major_axes: np.ndarray, omegas: np.ndarray) -> dict:
    """Fit log(omega) = slope * log(a) + intercept, comparing slope to -1.5.

    Inputs must be finite positive one-dimensional arrays of equal length,
    including at least two distinct semi-major axes.  Undefined R-squared for
    constant log frequencies is returned as NaN, with the same NaN in its
    derived metrics.  A perfect fit returns one_minus_r2=0 and its log10=-inf.
    The residual ratio is calculated directly as SSE/SST, without subtractive
    cancellation or clipping to make a near-perfect fit appear perfect.
    """
    if np.iscomplexobj(semi_major_axes) or np.iscomplexobj(omegas):
        raise ValueError("Semi-major axes and frequencies must be real-valued.")
    axes = np.asarray(semi_major_axes, dtype=np.float64)
    frequencies = np.asarray(omegas, dtype=np.float64)
    if axes.ndim != 1 or frequencies.ndim != 1 or axes.shape != frequencies.shape:
        raise ValueError("Semi-major axes and frequencies must be matching 1D arrays.")
    if axes.size < 2:
        raise ValueError("At least two observations are required for regression.")
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("Semi-major axes and frequencies must be finite.")
    if np.any(axes <= 0.0) or np.any(frequencies <= 0.0):
        raise ValueError("Semi-major axes and frequencies must be strictly positive.")
    x = np.log(axes)
    y = np.log(frequencies)
    if np.all(x == x[0]):
        raise ValueError("Regression requires distinct log semi-major axes.")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    if np.all(y == y[0]):
        slope = 0.0
        intercept = float(y[0])
        residual_ratio = float("nan")
    else:
        slope = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
        intercept = float(y.mean() - slope * x.mean())
        residual = y_centered - slope * x_centered
        residual_ratio = float(np.dot(residual, residual) / np.dot(y_centered, y_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_ratio = float(np.log10(residual_ratio))
    return {
        "slope": float(slope),
        "intercept": intercept,
        "slope_abs_error": float(abs(slope + 1.5)),
        "r2": float(1.0 - residual_ratio),
        "one_minus_r2": residual_ratio,
        "log10_one_minus_r2": log10_ratio,
    }
