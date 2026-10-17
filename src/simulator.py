"""Batched planar two-body dynamics in float64; time index zero is the initial state."""

from __future__ import annotations

from numbers import Integral

import numpy as np

MU = 1.0
DT = 0.01


def _positive_scalar(value: float, name: str) -> np.float64:
    value = np.float64(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _array(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values")
    return value


def _column(value: np.ndarray | float, n: int, name: str) -> np.ndarray:
    value = _array(value, name)
    if value.ndim == 0:
        return np.full((n, 1), value, dtype=np.float64)
    if value.shape != (n, 1):
        raise ValueError(f"{name} must be a scalar or have shape ({n}, 1)")
    return value


def _eccentric_anomaly(mean: np.ndarray, e: np.ndarray) -> np.ndarray:
    # Reduce phases before Newton to avoid loss of precision across many periods.
    mean = np.remainder(mean + np.pi, 2 * np.pi) - np.pi
    eccentricity = np.broadcast_to(e, mean.shape)
    anomaly = mean.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(8):
            anomaly -= (anomaly - eccentricity * np.sin(anomaly) - mean) / (
                1 - eccentricity * np.cos(anomaly)
            )
    residual = anomaly - eccentricity * np.sin(anomaly) - mean
    needs_fallback = ~np.isfinite(residual) | (np.abs(residual) > 8e-15)
    if np.any(needs_fallback):
        # Eight unguarded Newton steps are not sufficient for every e < 1.
        # A monotone bracket preserves the public domain for near-parabolic input.
        m = mean[needs_fallback]
        ecc = eccentricity[needs_fallback]
        low, high = m - ecc, m + ecc
        for _ in range(60):
            middle = (low + high) / 2
            below = middle - ecc * np.sin(middle) < m
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        anomaly[needs_fallback] = (low + high) / 2
    return anomaly


def generate_kepler_analytical(
    a: np.ndarray,
    e: np.ndarray,
    t: np.ndarray,
    mu: float = MU,
    omega_arg: np.ndarray | float = 0.0,
    M0: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return (N, T, 4) states [rx, ry, vx, vy] for bound Kepler orbits.

    ``a`` and ``e`` have shape (N, 1); ``t`` has shape (1, T).
    Negative times are allowed (useful for windows centered at an apsis).
    Eight Newton iterations are used, with a safeguarded fallback only when
    they do not converge for eccentricities near one. No integration drift.
    """
    mu = _positive_scalar(mu, "mu")
    a, e, t = _array(a, "a"), _array(e, "e"), _array(t, "t")
    if a.ndim != 2 or a.shape[1] != 1 or a.shape[0] == 0:
        raise ValueError("a must have nonempty shape (N, 1)")
    if e.shape != a.shape or np.any(e < 0) or np.any(e >= 1):
        raise ValueError("e must match a and satisfy 0 <= e < 1")
    if np.any(a <= 0):
        raise ValueError("a must be positive")
    if t.ndim != 2 or t.shape[0] != 1 or t.shape[1] == 0:
        raise ValueError("t must have nonempty shape (1, T)")
    omega_arg = _column(omega_arg, len(a), "omega_arg")
    M0 = _column(M0, len(a), "M0")
    n = np.sqrt(mu / a**3)
    anomaly = _eccentric_anomaly(n * t + M0, e)
    cos_e, sin_e = np.cos(anomaly), np.sin(anomaly)
    minor_ratio = np.sqrt(1 - e**2)
    radius = a * (1 - e * cos_e)
    x, y = a * (cos_e - e), a * minor_ratio * sin_e
    vx, vy = -(a**2 * n / radius) * sin_e, (a**2 * n / radius) * minor_ratio * cos_e
    cos_w, sin_w = np.cos(omega_arg), np.sin(omega_arg)
    return np.stack(
        (cos_w * x - sin_w * y, sin_w * x + cos_w * y,
         cos_w * vx - sin_w * vy, sin_w * vx + cos_w * vy), axis=-1
    ).astype(np.float64, copy=False)


def integrate_verlet(
    r0: np.ndarray, v0: np.ndarray, T_steps: int, dt: float = DT, mu: float = MU,
    substeps: int = 1,
) -> np.ndarray:
    """Velocity Verlet, with exactly T_steps samples including (r0, v0).

    Saved samples remain dt apart. Each interval contains ``substeps`` fixed
    symplectic updates of length dt/substeps; the default preserves Task 1.
    The singular gravitational origin is rejected rather than regularized.
    """
    mu, dt = _positive_scalar(mu, "mu"), _positive_scalar(dt, "dt")
    r, v = _array(r0, "r0").copy(), _array(v0, "v0").copy()
    if r.ndim != 2 or r.shape[1] != 2 or len(r) == 0 or v.shape != r.shape:
        raise ValueError("r0 and v0 must have matching nonempty shape (N, 2)")
    if isinstance(T_steps, bool) or not isinstance(T_steps, Integral) or T_steps < 1:
        raise ValueError("T_steps must be a positive integer")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    dt_sub = dt / substeps

    def acceleration(position: np.ndarray) -> np.ndarray:
        radius_squared = np.sum(position**2, axis=1, keepdims=True)
        if np.any(radius_squared == 0) or not np.all(np.isfinite(radius_squared)):
            raise ValueError("trajectory reached a singular or nonfinite position")
        return -mu * position / radius_squared**1.5

    trajectories = np.empty((len(r), T_steps, 4), dtype=np.float64)
    trajectories[:, 0, :2], trajectories[:, 0, 2:] = r, v
    acc = acceleration(r)
    for step in range(1, T_steps):
        for _ in range(substeps):
            half_v = v + 0.5 * dt_sub * acc
            r = r + dt_sub * half_v
            acc = acceleration(r)
            v = half_v + 0.5 * dt_sub * acc
        trajectories[:, step, :2], trajectories[:, step, 2:] = r, v
    return trajectories


def compute_orbital_invariants(trajectories: np.ndarray, mu: float = MU) -> dict:
    """Return invariant series and per-orbit maximum relative drift.

    Keys: energy, angular_momentum, a_eff (N,T); energy_drift and
    angular_momentum_drift (N,). Relative drift is NaN if the initial
    invariant is zero; a_eff is infinite for parabolic energy zero.
    """
    mu = _positive_scalar(mu, "mu")
    states = _array(trajectories, "trajectories")
    if states.ndim != 3 or states.shape[-1] != 4 or min(states.shape[:2]) < 1:
        raise ValueError("trajectories must have nonempty shape (N, T, 4)")
    r, v = states[..., :2], states[..., 2:]
    radius = np.linalg.norm(r, axis=-1)
    if np.any(radius == 0):
        raise ValueError("orbital invariants are singular at r=0")
    energy = np.sum(v**2, axis=-1) / 2 - mu / radius
    angular_momentum = r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]

    def drift(series: np.ndarray) -> np.ndarray:
        initial = np.abs(series[:, 0])
        result = np.full(len(series), np.nan, dtype=np.float64)
        np.divide(np.max(np.abs(series - series[:, :1]), axis=1), initial,
                  out=result, where=initial != 0)
        return result

    with np.errstate(divide="ignore", invalid="ignore"):
        a_eff = -mu / (2 * energy)
    return {"energy": energy, "angular_momentum": angular_momentum, "a_eff": a_eff,
            "energy_drift": drift(energy), "angular_momentum_drift": drift(angular_momentum)}
