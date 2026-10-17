"""Controlled JEPA losses with explicit Task 3.2 and legacy OPF reductions.

Default OPF uses population correlations after per-state RMS normalization;
the legacy option and isotropic loss use sample covariance, divisor N - 1.
All covariance losses require at least two samples. The isotropic loss is the explicit
mean/covariance surrogate in the task; it is not LeJEPA's characteristic-function
SIGReg implementation and does not establish Gaussianity.
"""

from numbers import Integral, Real
import math

import torch
from torch.nn import functional as F


def _sample_matrix(z: torch.Tensor) -> None:
    if not isinstance(z, torch.Tensor) or not z.is_floating_point():
        raise ValueError("z must be a real floating-point tensor")
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        raise ValueError("z must have shape (N >= 2, d >= 1)")


def ijepa_prediction_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Mean squared prediction error, with no gradient through the target."""
    if not isinstance(z_pred, torch.Tensor) or not isinstance(z_target, torch.Tensor):
        raise ValueError("predictions and targets must be tensors")
    if not z_pred.is_floating_point() or not z_target.is_floating_point():
        raise ValueError("predictions and targets must be real floating-point tensors")
    if z_pred.shape != z_target.shape or z_pred.numel() == 0:
        raise ValueError("predictions and targets must have equal, nonempty shapes")
    if z_pred.dtype != z_target.dtype or z_pred.device != z_target.device:
        raise ValueError("predictions and targets must have equal dtype and device")
    return F.mse_loss(z_pred, z_target.detach())


def opf_orthogonality_loss(z: torch.Tensor, K: int = 4,
                           eps: float = 1e-6, *, scale_invariant: bool = True,
                           reduction: str = "mean") -> torch.Tensor:
    """Task 3.2 per-state RMS normalization and mean off-block correlation².

    Population variance/correlation use divisor B. Standard deviations are
    floored at eps; constants contribute zero. No within-factor penalty.
    ``scale_invariant=False, reduction='legacy'`` exactly preserves the prior
    sample-standardized mean over ordered *factor pairs*, not all elements.
    ``sum`` gives the sum over ordered off-block elements before averaging.
    """
    _sample_matrix(z)
    if isinstance(K, bool) or not isinstance(K, Integral) or K < 2 or z.shape[1] % K:
        raise ValueError("K must be an integer >= 2 that divides the latent dimension")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive")
    if not isinstance(scale_invariant, bool) or reduction not in ("mean", "sum", "legacy"):
        raise ValueError("scale_invariant must be bool; reduction must be mean, sum or legacy")
    if scale_invariant:
        z = z / (z.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    centered = z.reshape(z.shape[0], int(K), z.shape[1] // K)
    centered = centered - centered.mean(dim=0, keepdim=True)
    divisor = z.shape[0] - 1 if reduction == "legacy" else z.shape[0]
    variance = centered.square().sum(dim=0, keepdim=True) / divisor
    # Clamp before sqrt: sqrt(0) followed by a clamp can produce NaN gradients
    # on constant coordinates through an intermediate infinite derivative.
    standardized = centered / variance.clamp_min(float(eps)**2).sqrt()
    total = z.new_zeros(())
    for j in range(int(K)):
        for k in range(j + 1, int(K)):
            cross = standardized[:, j].T @ standardized[:, k] / divisor
            total = total + cross.square().sum()
    if reduction == "legacy":
        return 2 * total / (K * (K - 1))
    if reduction == "sum":
        return 2 * total
    return 2 * total / (z.shape[1] ** 2 - K * (z.shape[1] // K) ** 2)


def sigreg_isotropic_loss(z: torch.Tensor) -> torch.Tensor:
    """Task-specified ``||mean||^2 + ||sample_covariance - I||_F^2``.

    This moment penalty alone does not exclude non-Gaussian representations.
    """
    _sample_matrix(z)
    mean = z.mean(dim=0)
    centered = z - mean
    covariance = centered.T @ centered / (z.shape[0] - 1)
    identity = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    return mean.square().sum() + (covariance - identity).square().sum()
