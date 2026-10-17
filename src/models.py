"""JEPA baselines with Task 3.1 odd encoders and action-angle predictors.

All constructors default to float32. For numerical spectral experiments pass
``dtype=torch.float64`` at construction: weights are then sampled and initialized
in float64, rather than quantized in float32 before conversion. These are the
controlled 1D architectures specified by this project, not exact reproductions
of the papers' complete training pipelines.

The MLP encoder defaults to ``odd_residual`` and trained models default to
``predictor_type='koopman'``. Select ``mlp_variant='legacy_gelu'`` and
``predictor_type='legacy_mlp'`` to reproduce the archived Task 2/3 architectures.
M0 always retains its frozen, unstructured MLP predictor.

Encoders consume (B, L, C); ``encode_sequence`` consumes (T, L, C), or optionally
(B, T, L, C), and preserves the leading dimensions. Rollouts consume (B, d) and
return (B, steps, d), excluding z0. They retain gradients, so callers conducting
evaluation should use ``eval()`` and ``torch.inference_mode()``.
"""

import copy
import math
from numbers import Integral, Real

import torch
from torch import nn

from .losses import ijepa_prediction_loss, opf_orthogonality_loss, sigreg_isotropic_loss


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")


def _weight(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _orthogonal_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 1e-3)
    elif isinstance(module, nn.MultiheadAttention):
        # The fused Q/K/V projection is a parameter, not an nn.Linear child.
        if module.in_proj_weight is not None:
            nn.init.orthogonal_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 1e-3)


def _sinusoidal(length, d, *, dtype, device):
    position = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
    inverse = torch.exp(torch.arange(0, d, 2, dtype=dtype, device=device)
                        * (-math.log(10000.0) / d))
    phase = position * inverse
    encoding = torch.zeros(length, d, dtype=dtype, device=device)
    encoding[:, 0::2] = phase.sin()
    encoding[:, 1::2] = phase[:, :d // 2].cos()
    return encoding


class OddResidualMLP(nn.Module):
    """Antipodally symmetric encoder with an initially small nonlinear branch.

    Oddness guarantees E(-x)=-E(x). It does not guarantee zero empirical mean on
    an incomplete arc: that conclusion also requires antipodally paired samples.
    Biases are absent, so optimization cannot break this symmetry.
    """

    def __init__(self, input_dim, d, *, dtype=torch.float32, device=None):
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.skip = nn.Linear(input_dim, d, bias=False, **factory)
        self.W1 = nn.Linear(input_dim, 128, bias=False, **factory)
        self.W2 = nn.Linear(128, 128, bias=False, **factory)
        self.W3 = nn.Linear(128, d, bias=False, **factory)
        for layer in (self.skip, self.W1, self.W2):
            nn.init.orthogonal_(layer.weight, gain=1.0)
        nn.init.orthogonal_(self.W3.weight, gain=0.15)

    def forward(self, x):
        return self.skip(x) + self.W3(torch.tanh(self.W2(torch.tanh(self.W1(x)))))


class SequenceEncoder(nn.Module):
    """Flattened linear/odd-residual MLP, or two-layer pre-norm transformer."""

    def __init__(self, L=16, C=4, d=64, arch_type="linear", *,
                 dtype=torch.float32, device=None, mlp_variant="odd_residual"):
        super().__init__()
        self.L, self.C, self.d = (_positive_int(value, name)
                                 for value, name in ((L, "L"), (C, "C"), (d, "d")))
        _dtype(dtype)
        if arch_type not in ("linear", "mlp", "transformer"):
            raise ValueError("arch_type must be 'linear', 'mlp', or 'transformer'")
        if mlp_variant not in ("odd_residual", "legacy_gelu"):
            raise ValueError("mlp_variant must be 'odd_residual' or 'legacy_gelu'")
        self.arch_type = arch_type
        self.mlp_variant = mlp_variant
        factory = {"dtype": dtype, "device": device}
        if arch_type == "linear":
            self.network = nn.Linear(self.L * self.C, self.d, bias=True, **factory)
        elif arch_type == "mlp":
            if mlp_variant == "odd_residual":
                self.network = OddResidualMLP(self.L * self.C, self.d, **factory)
            else:
                # The explicit archive option preserves Task 2/3 state keys,
                # bias initialization, and random-number consumption exactly.
                self.network = nn.Sequential(
                    nn.Linear(self.L * self.C, 128, **factory), nn.GELU(),
                    nn.Linear(128, 128, **factory), nn.GELU(),
                    nn.Linear(128, self.d, **factory),
                )
        else:
            if self.d % 4:
                raise ValueError("transformer latent dimension d must be divisible by 4")
            self.input_projection = nn.Linear(self.C, self.d, **factory)
            self.register_buffer("position_encoding", _sinusoidal(
                self.L, self.d, dtype=dtype, device=device))
            layer = nn.TransformerEncoderLayer(
                d_model=self.d, nhead=4, dim_feedforward=128,
                batch_first=True, norm_first=True, **factory)
            self.network = nn.TransformerEncoder(layer, num_layers=2,
                                                 enable_nested_tensor=False)
        # Applied after the transformer clones its layers, giving each its own
        # orthogonal draw instead of leaving identically initialized clones.
        if arch_type != "mlp" or mlp_variant == "legacy_gelu":
            self.apply(_orthogonal_init)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (self.L, self.C):
            raise ValueError(f"x must have shape (B, {self.L}, {self.C})")
        if self.arch_type == "transformer":
            tokens = self.input_projection(x) + self.position_encoding.unsqueeze(0)
            return self.network(tokens).mean(dim=1)
        return self.network(x.flatten(start_dim=1))


class HorizonPredictor(nn.Module):
    """Residual MLP ``z + MLP(z + embedding[k])`` for k in 1..H_max."""

    def __init__(self, d=64, H_max=64, *, dtype=torch.float32, device=None):
        super().__init__()
        self.d, self.H_max = _positive_int(d, "d"), _positive_int(H_max, "H_max")
        self.supports_extrapolation = False
        _dtype(dtype)
        factory = {"dtype": dtype, "device": device}
        self.pos_emb = nn.Parameter(torch.empty(self.H_max + 1, self.d, **factory))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(self.d, 128, **factory), nn.GELU(),
                                 nn.Linear(128, self.d, **factory))
        self.apply(_orthogonal_init)

    def forward(self, z_t, k=1):
        if not isinstance(z_t, torch.Tensor) or z_t.ndim != 2 or z_t.shape[1] != self.d:
            raise ValueError(f"z_t must have shape (B, {self.d})")
        if isinstance(k, bool):
            raise ValueError("k must be an integer or integer tensor")
        if isinstance(k, Integral):
            if not 1 <= k <= self.H_max:
                raise ValueError("k must lie in 1..H_max")
            embedding = self.pos_emb[int(k)]
        elif isinstance(k, torch.Tensor):
            if k.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("k tensor must contain integer horizon indices")
            if k.ndim not in (0, 1) or (k.ndim == 1 and k.shape[0] != z_t.shape[0]):
                raise ValueError("k must be scalar or have shape (B,)")
            if torch.any((k < 1) | (k > self.H_max)):
                raise ValueError("all k values must lie in 1..H_max")
            embedding = self.pos_emb[k.to(device=self.pos_emb.device, dtype=torch.long)]
        else:
            raise ValueError("k must be an integer or integer tensor")
        return z_t + self.mlp(z_t + embedding)


class ActionAngleKoopmanPredictor(nn.Module):
    """Norm-preserving action-dependent planar rotations in a learned SO(d) basis.

    All frequencies are learned from latent actions; this module receives no
    orbital radius, physical frequency, simulator metadata, or Kepler formula.
    H_max records the training horizon, while the same operator supports every
    positive integer horizon, including extrapolation beyond 512.

    Reuse ``orthogonal_basis()`` with ``basis=Q`` throughout one loss graph or
    rollout. This evaluates matrix_exp once, retains its gradient, and avoids
    retaining a stale graph or stale basis after optimizer updates.
    """

    supports_extrapolation = True

    def __init__(self, d=64, H_max=64, *, dt_pred=0.04, K=1,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.d, self.H_max, self.K = (_positive_int(value, name) for value, name in
                                      ((d, "d"), (H_max, "H_max"), (K, "K")))
        _dtype(dtype)
        if self.d % self.K or (self.d // self.K) % 2:
            raise ValueError("K must divide d and each factor dimension must be even")
        if (isinstance(dt_pred, bool) or not isinstance(dt_pred, Real)
                or not math.isfinite(dt_pred) or dt_pred <= 0):
            raise ValueError("dt_pred must be finite and positive")
        self.dt_pred, self.d_f = float(dt_pred), self.d // self.K
        factory = {"dtype": dtype, "device": device}
        shape = (self.d, self.d) if self.K == 1 else (self.K, self.d_f, self.d_f)
        self.S = nn.Parameter(torch.zeros(shape, **factory))
        self.rate_net = nn.Sequential(
            nn.Linear(self.d // 2, 128, **factory), nn.GELU(),
            nn.Linear(128, 128, **factory), nn.GELU(),
            nn.Linear(128, self.d // 2, **factory),
        )
        for layer in self.rate_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def orthogonal_basis(self):
        """Q=exp(S-S.T); M2 uses independent contiguous factor blocks."""
        Q = torch.matrix_exp(self.S - self.S.transpose(-1, -2))
        return Q if self.K == 1 else torch.block_diag(*Q.unbind(0))

    def _check_state(self, z):
        if (not isinstance(z, torch.Tensor) or not z.is_floating_point()
                or z.ndim != 2 or z.shape[1] != self.d):
            raise ValueError(f"z_t must have shape (B, {self.d}) and floating dtype")

    def _basis(self, z, basis):
        if basis is None:
            return self.orthogonal_basis()
        if (not isinstance(basis, torch.Tensor) or basis.shape != (self.d, self.d)
                or basis.dtype != z.dtype or basis.device != z.device):
            raise ValueError("basis must have shape (d,d) and match z dtype/device")
        return basis

    def plane_log_actions(self, z, *, basis=None):
        self._check_state(z)
        Q = self._basis(z, basis)
        u = (z @ Q.T).reshape(z.shape[0], self.d // 2, 2)
        return torch.log(torch.sqrt(u.square().sum(dim=-1) + 1e-8) + 1e-6)

    def angular_rates(self, z, *, basis=None):
        return torch.nn.functional.softplus(self.rate_net(self.plane_log_actions(z, basis=basis)))

    def _state_and_rates(self, z, basis):
        self._check_state(z)
        Q = self._basis(z, basis)
        u = (z @ Q.T).reshape(z.shape[0], self.d // 2, 2)
        actions = torch.log(torch.sqrt(u.square().sum(dim=-1) + 1e-8) + 1e-6)
        rates = torch.nn.functional.softplus(self.rate_net(actions))
        return Q, u, rates

    @staticmethod
    def _rotate(u, angle):
        cosine, sine = angle.cos(), angle.sin()
        x, y = u[..., 0], u[..., 1]
        return torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)

    def _horizons(self, k, device, *, batch_size=None, shared=False):
        if isinstance(k, bool):
            raise ValueError("horizons must be positive integers")
        if isinstance(k, Integral):
            if k < 1 or shared:
                raise ValueError("horizons must be positive integer tensor when shared")
            return torch.tensor(k, device=device, dtype=torch.long)
        if (not isinstance(k, torch.Tensor) or k.dtype not in
                (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or k.ndim not in ((1,) if shared else (0, 1))):
            raise ValueError("horizons must be an integer scalar or one-dimensional integer tensor")
        if not shared and k.ndim == 1 and k.shape[0] != batch_size:
            raise ValueError("per-batch horizons must have shape (B,)")
        if torch.any(k < 1):
            raise ValueError("horizons must be positive")
        return k.to(device=device, dtype=torch.long)

    def forward(self, z_t, k=1, *, basis=None):
        Q, u, rates = self._state_and_rates(z_t, basis)
        horizons = self._horizons(k, z_t.device, batch_size=z_t.shape[0])
        if horizons.ndim:
            horizons = horizons[:, None]
        angle = rates * horizons * self.dt_pred
        return self._rotate(u, angle).flatten(start_dim=1) @ Q

    def predict_horizons(self, z_t, horizons, *, basis=None):
        """Evaluate a shared 1D integer horizon vector, returning (B,H,d)."""
        Q, u, rates = self._state_and_rates(z_t, basis)
        horizons = self._horizons(horizons, z_t.device, shared=True)
        angle = rates[:, None, :] * horizons[None, :, None] * self.dt_pred
        rotated = self._rotate(u[:, None, :, :], angle)
        return rotated.flatten(start_dim=2) @ Q


class _WorldModel(nn.Module):
    """Common T1/T3/T4 evaluation API; no training or inference mode changes."""

    @property
    def encoder(self):
        """Alias for the context encoder without duplicate state-dict keys."""
        return self.context_encoder

    def forward(self, x_c, k=1):
        return self.predictor(self.context_encoder(x_c), k)

    def encode_sequence(self, traj_windows, batch_size=None):
        if not isinstance(traj_windows, torch.Tensor) or traj_windows.ndim not in (3, 4):
            raise ValueError("traj_windows must have shape (T,L,C) or (B,T,L,C)")
        encoder = self.context_encoder
        if traj_windows.shape[-2:] != (encoder.L, encoder.C):
            raise ValueError("window dimensions must match the context encoder")
        leading = traj_windows.shape[:-2]
        flattened = traj_windows.reshape(-1, encoder.L, encoder.C)
        if batch_size is None:
            encoded = encoder(flattened)
        else:
            batch_size = _positive_int(batch_size, "batch_size")
            if flattened.shape[0] == 0:
                encoded = flattened.new_empty((0, encoder.d))
            else:
                encoded = torch.cat([encoder(chunk) for chunk in flattened.split(batch_size)], dim=0)
        return encoded.reshape(*leading, encoder.d)

    def _rollout_steps(self, z0, steps, direct=False):
        if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or z0.shape[1] != self.context_encoder.d:
            raise ValueError(f"z0 must have shape (B, {self.context_encoder.d})")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if direct and steps > self.predictor.H_max and not getattr(self.predictor, "supports_extrapolation", False):
            raise ValueError("direct rollout exceeds H_max")
        return int(steps)

    def rollout_autoregressive(self, z0, steps, *, basis=None):
        """T3: repeatedly apply horizon 1, preserving the computation graph."""
        steps = self._rollout_steps(z0, steps)
        structured = isinstance(self.predictor, ActionAngleKoopmanPredictor)
        if structured and basis is None:
            basis = self.predictor.orthogonal_basis()
        if not structured and basis is not None:
            raise ValueError("basis is available only for the Koopman predictor")
        state, outputs = z0, []
        for _ in range(steps):
            state = self.predictor(state, 1, basis=basis) if structured else self.predictor(state, 1)
            outputs.append(state)
        return torch.stack(outputs, dim=1) if outputs else z0.new_empty((z0.shape[0], 0, z0.shape[1]))

    def rollout_direct(self, z0, steps, *, basis=None):
        """T4: predict horizons 1..steps from the same initial state."""
        steps = self._rollout_steps(z0, steps, direct=True)
        if isinstance(self.predictor, ActionAngleKoopmanPredictor):
            if steps == 0:
                return z0.new_empty((z0.shape[0], 0, z0.shape[1]))
            horizons = torch.arange(1, steps + 1, device=z0.device)
            return self.predictor.predict_horizons(z0, horizons, basis=basis)
        if basis is not None:
            raise ValueError("basis is available only for the Koopman predictor")
        outputs = [self.predictor(z0, k) for k in range(1, steps + 1)]
        return torch.stack(outputs, dim=1) if outputs else z0.new_empty((z0.shape[0], 0, z0.shape[1]))


class M0_RandomJEPA(_WorldModel):
    """Untrained frozen encoder/predictor, including the Exp1 transformer."""

    def __init__(self, L=16, C=4, d=64, H_max=64, arch_type="linear", *,
                 dtype=torch.float32, device=None, mlp_variant="odd_residual"):
        super().__init__()
        self.context_encoder = SequenceEncoder(L, C, d, arch_type, dtype=dtype, device=device,
                                               mlp_variant=mlp_variant)
        self.predictor = HorizonPredictor(d, H_max, dtype=dtype, device=device)
        self.requires_grad_(False)


class M0_Koopman(_WorldModel):
    """Frozen odd encoder and action-angle predictor, initialized as M1.

    Constructed with the same seed as M1, the online encoder and predictor
    start with identical parameters. No target, optimizer or training is used.
    """

    def __init__(self, L=16, C=4, d=64, H_max=64, arch_type="mlp", *,
                 dtype=torch.float32, device=None, mlp_variant="odd_residual", dt_pred=0.04):
        super().__init__()
        self.context_encoder = SequenceEncoder(L, C, d, arch_type, dtype=dtype,
                                               device=device, mlp_variant=mlp_variant)
        self.predictor = ActionAngleKoopmanPredictor(d, H_max, dt_pred=dt_pred,
                                                    dtype=dtype, device=device)
        self.requires_grad_(False)


class M1_IJEPA1D(_WorldModel):
    """Context/EMA-target pair with a shared action-angle Koopman predictor."""

    def __init__(self, L=16, C=4, d=64, H_max=64, arch_type="mlp", *,
                 dtype=torch.float32, device=None, mlp_variant="odd_residual",
                 predictor_type="koopman", dt_pred=0.04):
        super().__init__()
        if predictor_type not in ("koopman", "legacy_mlp"):
            raise ValueError("predictor_type must be 'koopman' or 'legacy_mlp'")
        self.predictor_type = predictor_type
        self.context_encoder = SequenceEncoder(L, C, d, arch_type, dtype=dtype, device=device,
                                               mlp_variant=mlp_variant)
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = (ActionAngleKoopmanPredictor(d, H_max, dt_pred=dt_pred,
                                                     dtype=dtype, device=device)
                          if predictor_type == "koopman" else
                          HorizonPredictor(d, H_max, dtype=dtype, device=device))

    def train(self, mode=True):
        super().train(mode)
        # Target features are deterministic even when the context has dropout.
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_ema(self, momentum=0.996):
        if (isinstance(momentum, bool) or not isinstance(momentum, Real)
                or not math.isfinite(momentum) or not 0 <= momentum <= 1):
            raise ValueError("momentum must be finite and lie in [0,1]")
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.mul_(momentum).add_(context, alpha=1 - momentum)
        for target, context in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            if target.is_floating_point():
                target.mul_(momentum).add_(context, alpha=1 - momentum)
            else:
                target.copy_(context)

    def compute_loss(self, z_pred, z_target, *, z_context=None):
        """Prediction-only objective; z_context is used by regularized subclasses."""
        return ijepa_prediction_loss(z_pred, z_target)

    def training_loss(self, x_c, x_target, k=1, **loss_kwargs):
        """Encode a batch of context/target windows and evaluate its objective."""
        z_context = self.context_encoder(x_c)
        z_pred = self.predictor(z_context, k)
        with torch.no_grad():
            z_target = self.target_encoder(x_target)
        return self.compute_loss(z_pred, z_target, z_context=z_context, **loss_kwargs)


class _FactorizedPredictor(nn.Module):
    def __init__(self, d, K, H_max, *, dtype, device):
        super().__init__()
        self.d, self.K, self.H_max = d, K, _positive_int(H_max, "H_max")
        self.predictors = nn.ModuleList([
            HorizonPredictor(d // K, H_max, dtype=dtype, device=device) for _ in range(K)
        ])

    def forward(self, z_t, k=1):
        if not isinstance(z_t, torch.Tensor) or z_t.ndim != 2 or z_t.shape[1] != self.d:
            raise ValueError(f"z_t must have shape (B, {self.d})")
        return torch.cat([predictor(factor, k) for predictor, factor in
                          zip(self.predictors, z_t.split(self.d // self.K, dim=-1))], dim=-1)


class M2_JEPA_OPF(M1_IJEPA1D):
    """Koopman basis with contiguous orthogonal factor blocks.

    The rate network may read all invariant plane actions; the orthogonal basis
    cannot mix factors. The legacy option retains independent MLP predictors.
    """

    def __init__(self, L=16, C=4, d=64, H_max=64, K=4, arch_type="mlp", *,
                 dtype=torch.float32, device=None, mlp_variant="odd_residual",
                 predictor_type="koopman", dt_pred=0.04):
        K = _positive_int(K, "K")
        d = _positive_int(d, "d")
        if d % K:
            raise ValueError("K must divide d")
        super().__init__(L, C, d, H_max, arch_type, dtype=dtype, device=device,
                         mlp_variant=mlp_variant, predictor_type=predictor_type, dt_pred=dt_pred)
        self.K, self.d_f = K, d // K
        self.predictor = (ActionAngleKoopmanPredictor(d, H_max, K=K, dt_pred=dt_pred,
                                                     dtype=dtype, device=device)
                          if predictor_type == "koopman" else
                          _FactorizedPredictor(d, K, H_max, dtype=dtype, device=device))

    @property
    def predictors(self):
        """Independent predictor modules of the explicit legacy architecture."""
        return self.predictor.predictors

    def get_active_factor(self, Z_seq):
        """Return the index with maximum trace of temporal sample covariance.

        Accept (T,d) or (B,T,d); for batched sequences average the per-sequence
        temporal variances, without mixing between-orbit offsets into variance.
        Ties select the first index, consistently with torch.argmax.
        """
        if (not isinstance(Z_seq, torch.Tensor) or not Z_seq.is_floating_point()
                or Z_seq.ndim not in (2, 3) or Z_seq.shape[-1] != self.context_encoder.d
                or Z_seq.shape[-2] < 2):
            raise ValueError("Z_seq must have shape (T>=2,d) or (B,T>=2,d)")
        factors = Z_seq.reshape(*Z_seq.shape[:-1], self.K, self.d_f)
        scores = factors.var(dim=-3, correction=1).sum(dim=-1)
        if scores.ndim == 2:
            scores = scores.mean(dim=0)
        return int(scores.argmax().item())

    def compute_loss(self, z_pred, z_target, *, z_context=None, orthogonality_weight=1.0):
        weight = _weight(orthogonality_weight, "orthogonality_weight")
        prediction = ijepa_prediction_loss(z_pred, z_target)
        if weight == 0:
            return prediction
        states = z_pred if z_context is None else z_context
        return prediction + weight * opf_orthogonality_loss(
            states, self.K, scale_invariant=self.predictor_type != "legacy_mlp",
            reduction="legacy" if self.predictor_type == "legacy_mlp" else "mean")


class M3_IJEPA_SIGReg(M1_IJEPA1D):
    """I-JEPA plus the task's explicit isotropic mean/covariance penalty.

    This uses the requested formula rather than LeJEPA's full SIGReg algorithm.
    ``training_loss`` regularizes context embeddings; ``compute_loss`` can be
    supplied z_context, or by default regularizes z_pred. Weight zero disables it.
    Task 3 can additionally enable pointwise, non-affine sphere normalization.
    Its fixed radius is sqrt(d), i.e. unit coordinate RMS, compatible with the
    covariance-I objective. This never uses a whole orbit or future samples.
    """

    def __init__(self, *args, normalize_embeddings=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_embeddings = bool(normalize_embeddings)
        if self.normalize_embeddings:
            self.context_encoder = _SphereEncoder(self.context_encoder)
            self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
            self.target_encoder.eval()
            # Koopman already preserves the input norm. Re-centering generated
            # states after each rotation would destroy its invariant actions
            # and semigroup, so only legacy M3 applies that archive constraint.
            if self.predictor_type == "legacy_mlp":
                self.predictor = _SpherePredictor(self.predictor)

    def compute_loss(self, z_pred, z_target, *, z_context=None, sigreg_weight=1.0):
        weight = _weight(sigreg_weight, "sigreg_weight")
        prediction = ijepa_prediction_loss(z_pred, z_target)
        if weight == 0:
            return prediction
        states = z_pred if z_context is None else z_context
        return prediction + weight * sigreg_isotropic_loss(states)


class _SphereEncoder(nn.Module):
    """Pointwise LayerNorm plus fixed-RMS rescaling; no orbit-level fitting."""

    def __init__(self, encoder):
        super().__init__()
        self.base_encoder = encoder
        self.L, self.C, self.d, self.arch_type = encoder.L, encoder.C, encoder.d, encoder.arch_type

    def forward(self, x):
        raw = self.base_encoder(x)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        normalized = torch.nn.functional.normalize(centered, p=2, dim=-1, eps=1e-12)
        return normalized * math.sqrt(self.d)


class _SpherePredictor(nn.Module):
    """Apply the same pointwise fixed-RMS constraint to generated M3 states."""

    def __init__(self, predictor):
        super().__init__()
        self.base_predictor = predictor
        self.d, self.H_max = predictor.d, predictor.H_max

    def forward(self, z_t, k=1):
        raw = self.base_predictor(z_t, k)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        return torch.nn.functional.normalize(centered, p=2, dim=-1, eps=1e-12) * math.sqrt(self.d)
