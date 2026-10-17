"""Deterministic orbit-level splits and on-demand overlapping PyTorch windows.

Builders return a dict {trajectories, meta, dt, split}. Elliptical test splits
return {perihelion_init: batch, aphelion_init: batch} with matched orbital
parameters. All generated states and floating metadata are float64.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulator import DT, MU, generate_kepler_analytical, integrate_verlet

CONTEXT_LENGTH = 16


def _axes(split: str) -> np.ndarray:
    if split == "train":
        return np.geomspace(0.5, 5.0, 200, dtype=np.float64)
    if split == "test_id":
        # Half a train-grid step inward at both ends: disjoint orbit radii.
        half_step = (5.0 / 0.5) ** (0.5 / 199)
        return np.geomspace(0.5 * half_step, 5.0 / half_step, 50, dtype=np.float64)
    if split == "test_ood":
        return np.concatenate((np.geomspace(0.25, 0.45, 25), np.geomspace(5.5, 8.0, 25)))
    raise ValueError("split must be train, test_id, or test_ood")


def _rng(seed: int, split: str, kind: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    split_index = {"train": 0, "test_id": 1, "test_ood": 2}[split]
    return np.random.default_rng(np.random.SeedSequence([seed, split_index, kind]))


def _analytical_batch(a: np.ndarray, e: np.ndarray, omega_arg: np.ndarray,
                      M0: np.ndarray, split: str) -> dict:
    time = np.arange(1024, dtype=np.float64)[None, :] * DT
    trajectories = generate_kepler_analytical(a[:, None], e[:, None], time,
                                              omega_arg=omega_arg[:, None], M0=M0[:, None])
    return {"trajectories": trajectories, "dt": DT, "split": split,
            "meta": {"a": a.copy(), "e": e.copy(), "omega_arg": omega_arg.copy(),
                     "M0": M0.copy(), "omega_true": np.sqrt(MU / a**3)}}


def build_dataset_circ(split: str, seed: int = 42) -> dict:
    """Circular orbits, T=1024, dt=.01, with randomized initial mean anomaly."""
    a = _axes(split)
    rng = _rng(seed, split, 0)
    return _analytical_batch(a, np.zeros(len(a)), np.zeros(len(a)),
                             rng.uniform(0, 2 * np.pi, len(a)), split)


def build_dataset_ellip(split: str, seed: int = 42) -> dict:
    """Ellipses with e in [.2,.6]; test contexts centered at each apsis.

    For even L=16 the center is between samples 7 and 8, at t=.075.
    M0 = M_apsis - n*.075 makes the continuous orbit's apsis exact there.
    Train phases are uniform. Test variants share a, e, and omega_arg.
    """
    a = _axes(split)
    rng = _rng(seed, split, 1)
    e = rng.uniform(0.2, 0.6, len(a))
    omega_arg = rng.uniform(0, 2 * np.pi, len(a))
    if split == "train":
        return _analytical_batch(a, e, omega_arg, rng.uniform(0, 2 * np.pi, len(a)), split)
    phase_shift = np.sqrt(MU / a**3) * ((CONTEXT_LENGTH - 1) * DT / 2)
    return {"perihelion_init": _analytical_batch(a, e, omega_arg, -phase_shift, split),
            "aphelion_init": _analytical_batch(a, e, omega_arg, np.pi - phase_shift, split)}


def build_dataset_counterfactual(seed: int = 42) -> dict:
    """Velocity-perturbed bound orbits, T=2048; meta['a'] is initial radius.

    It is NOT the perturbed orbit's semi-major axis: use meta['a_true'].
    The initial velocity is transverse to the initial position.
    """
    a = np.geomspace(0.5, 5.0, 50, dtype=np.float64)
    rng = _rng(seed, "test_id", 2)
    alpha = rng.uniform(0.75, 1.15, len(a))
    while np.any(alpha == 1.0):
        mask = alpha == 1.0
        alpha[mask] = rng.uniform(0.75, 1.15, np.count_nonzero(mask))
    r0, v0 = np.zeros((len(a), 2)), np.zeros((len(a), 2))
    r0[:, 0], v0[:, 1] = a, alpha * np.sqrt(MU / a)
    a_true = a / (2 - alpha**2)
    return {"trajectories": integrate_verlet(r0, v0, 2048), "dt": DT, "split": "counterfactual",
            "meta": {"a": a, "initial_radius": a.copy(), "alpha": alpha,
                     "a_true": a_true, "e": np.abs(alpha**2 - 1),
                     "omega_init": alpha * np.sqrt(MU) * a**-1.5,
                     "omega_true": np.sqrt(MU) * a_true**-1.5}}


class KeplerWindowDataset(Dataset):
    """All valid windows; target[k-1] starts k samples after the context.

    Windows never cross orbit boundaries. Only the requested item is cast to
    float32; no (N,T,H,L,C) target array is materialized. Numeric per-orbit
    metadata is optional. Context/target tensors own their memory so an in-place
    transform cannot modify the stored float64 ground truth.
    """

    def __init__(self, trajectories: np.ndarray, L: int = 16, H: int = 128,
                 channels: int = 4, meta: Mapping[str, np.ndarray] | None = None):
        if np.iscomplexobj(trajectories):
            raise ValueError("trajectories must be real")
        states = np.asarray(trajectories)
        if states.ndim != 3 or states.shape[0] < 1 or states.shape[2] not in (2, 4):
            raise ValueError("trajectories must have nonempty shape (N,T,2) or (N,T,4)")
        if not np.all(np.isfinite(states)):
            raise ValueError("trajectories must be finite")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (L, H)):
            raise ValueError("L and H must be positive integers")
        if isinstance(channels, bool) or channels not in (2, 4) or channels > states.shape[2]:
            raise ValueError("channels must be 2 or 4, available in trajectories")
        self.trajectories = states
        self.L, self.H, self.channels = int(L), int(H), int(channels)
        self.windows_per_orbit = states.shape[1] - self.L - self.H + 1
        if self.windows_per_orbit < 1:
            raise ValueError("trajectory must contain at least L + H samples")
        self.meta = {}
        for key, value in (meta or {}).items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (len(states),) or not np.all(np.isfinite(array)):
                raise ValueError(f"meta[{key!r}] must have finite shape (N,)")
            self.meta[key] = array.copy()

    def __len__(self) -> int:
        return len(self.trajectories) * self.windows_per_orbit

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("window index out of range")
        orbit, start = divmod(index, self.windows_per_orbit)
        trajectory = self.trajectories[orbit, :, :self.channels]
        context = np.array(trajectory[start:start + self.L], dtype=np.float32, copy=True)
        indices = start + np.arange(1, self.H + 1)[:, None] + np.arange(self.L)[None, :]
        targets = np.array(trajectory[indices], dtype=np.float32, copy=True)
        metadata = {key: torch.tensor(value[orbit], dtype=torch.float32)
                    for key, value in self.meta.items()}
        metadata.update(orbit_index=orbit, start_index=start, dt=torch.tensor(DT, dtype=torch.float32))
        return torch.from_numpy(context), torch.from_numpy(targets), metadata
