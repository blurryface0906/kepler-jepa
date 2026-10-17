"""Task 3 and Task 3.1 GPU training of the controlled JEPA world models.

The production protocol is fixed at 2,000 steps, batches of 256, 64 direct
horizons and 16 autoregressive steps. There is no automatic CPU fallback.
``smoke_test=True`` explicitly permits smaller configurations for engineering
tests; such a run must never be reported as the production experiment.

Squared L2 prediction errors sum over latent coordinates, then average over
batch and horizons, as in the task's formula. The variance barrier uses the
online encoder at the context and its next macro-step, so both its state and
velocity terms can train the encoder. M3 replaces that barrier with the stated
isotropic penalty (beta=1); M2's optional sparsity term is disabled by default.

``protocol="task3_1"`` selects the remediation objective without modifying the
archived Task 3 objective. Direct squared errors are normalized per orbit by
the detached mean squared L2 norm of their EMA state targets. The prior AR
state and increment losses retain summed-coordinate L2 errors. M1/M2 replace the
batch barrier with the specified 16-macro-step displacement and RMS matching
penalty, at weight 1. M3 retains its fixed-norm ablation and global isotropic
penalty, without the temporal/scale term. No orbit radii or physical frequencies
are supplied to these losses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import random
import time
from typing import Callable, Mapping

import numpy as np
import torch

from .losses import opf_orthogonality_loss, sigreg_isotropic_loss
from .models import M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256
    context_length: int = 16
    delta: int = 4
    horizon: int = 64
    ar_steps: int = 16
    device: str = "cuda"
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    ema_start: float = 0.99
    ema_end: float = 0.999
    velocity_weight: float = 1.0
    variance_weight: float = 0.1
    state_std_floor: float = 0.5
    velocity_std_floor: float = 0.05
    variance_epsilon: float = 1e-6
    orthogonality_weight: float = 0.2
    sparsity_weight: float = 0.0
    sigreg_weight: float = 1.0
    log_every: int = 20
    checkpoint_every: int = 100
    smoke_test: bool = False
    protocol: str = "task3"
    temporal_lag: int = 16
    temporal_displacement_fraction: float = 0.8
    temporal_scale_weight: float = 0.2
    temporal_weight: float = 1.0
    relative_denominator_epsilon: float = 1e-4
    log_scale_epsilon: float = 1e-6
    log_scale_weight: float = 0.5
    factor_scale_weight: float = 0.25
    coherence_weight: float = 0.1
    ellip_variant: str = "inv"
    frequency_invariance_weight: float = 2.0

    def __post_init__(self):
        for name in ("steps", "batch_size", "context_length", "delta", "horizon",
                     "ar_steps", "log_every", "checkpoint_every", "temporal_lag"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2 or self.ar_steps > self.horizon:
            raise ValueError("batch_size must be >=2 and ar_steps must not exceed horizon")
        for name in ("learning_rate", "minimum_learning_rate", "gradient_clip",
                     "variance_epsilon", "relative_denominator_epsilon", "log_scale_epsilon"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "velocity_weight", "variance_weight",
                     "state_std_floor", "velocity_std_floor", "orthogonality_weight",
                     "sparsity_weight", "sigreg_weight", "temporal_displacement_fraction",
                     "temporal_scale_weight", "temporal_weight", "factor_scale_weight", "coherence_weight", "log_scale_weight", "frequency_invariance_weight"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate must not exceed learning_rate")
        if (any(isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) for value in (self.ema_start, self.ema_end))
                or not 0 <= self.ema_start <= self.ema_end <= 1):
            raise ValueError("EMA schedule must satisfy 0 <= start <= end <= 1")
        if not isinstance(self.smoke_test, bool):
            raise ValueError("smoke_test must be a boolean")
        if self.protocol not in ("task3", "task3_1", "task3_2", "task4_ellip"):
            raise ValueError("unsupported training protocol")
        if self.protocol == 'task4_ellip' and (self.ellip_variant not in ('short','inv','sigreg') or self.temporal_lag > self.horizon):
            raise ValueError('Invalid Task4 variant or temporal lag beyond horizon')
        if self.protocol in ("task3_1", "task3_2") and self.temporal_lag > self.horizon:
            raise ValueError("Task 3.1 temporal_lag must not exceed horizon")
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("device must name a torch device") from error
        if not self.smoke_test:
            fixed = {"steps": 2000, "batch_size": 256, "context_length": 16,
                     "horizon": 64, "ar_steps": 16}
            if self.protocol == "task4_ellip":
                if self.ellip_variant not in ("short", "inv", "sigreg"):
                    raise ValueError("Unknown ellipse variant")
                fixed["horizon"] = 32 if self.ellip_variant == "short" else 256
                if self.delta != 4:
                    raise ValueError("Task4 requires delta=4")
            if any(getattr(self, name) != value for name, value in fixed.items()):
                raise ValueError("production protocol cannot be shortened; use smoke_test=True for unit tests")
            if self.delta not in (2, 4) or device.type != "cuda":
                raise ValueError("production training requires CUDA and delta=2 or 4")
            if self.protocol in ("task3_1", "task3_2") and (self.delta != 4 or self.temporal_lag != 16):
                raise ValueError("Task 3.1 production requires delta=4 and temporal_lag=16")


def batch_variance_loss(z: torch.Tensor, delta_z: torch.Tensor,
                        config: TrainConfig) -> torch.Tensor:
    """State/increment standard-deviation barrier across the entire batch.

    Population variance (divisor B) matches Var_B; no per-orbit rescaling or
    normalization is performed here. ``delta_z`` must remain on the online
    encoder's computation graph.
    """
    if (z.ndim != 2 or z.shape[0] < 2 or delta_z.shape != z.shape
            or z.dtype != delta_z.dtype or z.device != delta_z.device):
        raise ValueError("z and delta_z must have equal (B>=2,d) shape, dtype and device")
    state_std = (z.var(dim=0, correction=0) + config.variance_epsilon).sqrt()
    velocity_std = (delta_z.var(dim=0, correction=0) + config.variance_epsilon).sqrt()
    return ((config.state_std_floor - state_std).clamp_min(0).mean()
            + (config.velocity_std_floor - velocity_std).clamp_min(0).mean())


def _squared_l2_mean(prediction, target):
    return (prediction - target.detach()).square().sum(dim=-1).mean()


def _rms_per_orbit(value: torch.Tensor) -> torch.Tensor:
    """Coordinate RMS over all non-batch axes, with finite zero gradients."""
    flattened = value.flatten(start_dim=1)
    return torch.linalg.vector_norm(flattened, dim=1) / math.sqrt(flattened.shape[1])


def temporal_displacement_loss(x0: torch.Tensor, x_lag: torch.Tensor,
                               z0: torch.Tensor, z_lag: torch.Tensor,
                               config: TrainConfig) -> torch.Tensor:
    """Exact intra-orbit displacement + input/latent RMS matching formula.

    Input window RMS uses all L*C entries; latent RMS uses all d coordinates.
    The raw input and its displacement are detached targets. Both online
    embeddings retain gradients; no batch variance or cross-orbit statistic
    enters this term. ``torch.linalg.vector_norm`` gives a finite subgradient
    at exactly zero, avoiding sqrt-of-zero backward NaNs.
    """
    if (not isinstance(x0, torch.Tensor) or not isinstance(x_lag, torch.Tensor)
            or not isinstance(z0, torch.Tensor) or not isinstance(z_lag, torch.Tensor)
            or x0.ndim != 3 or x_lag.shape != x0.shape or z0.ndim != 2
            or z_lag.shape != z0.shape or z0.shape[0] != x0.shape[0]
            or x0.shape[0] < 1 or not all(value.is_floating_point()
                                        for value in (x0, x_lag, z0, z_lag))
            or any(value.dtype != z0.dtype or value.device != z0.device
                   for value in (x0, x_lag, z_lag))):
        raise ValueError("matching floating input windows and latent batches are required")
    input_displacement = _rms_per_orbit(x0.detach() - x_lag.detach())
    latent_displacement = _rms_per_orbit(z0 - z_lag)
    phase = (config.temporal_displacement_fraction * input_displacement
             - latent_displacement).clamp_min(0)
    if config.protocol == "task3_2":
        scale = (torch.log(_rms_per_orbit(z0) + config.log_scale_epsilon)
                 - torch.log(_rms_per_orbit(x0.detach()) + config.log_scale_epsilon)).square()
        return (phase + config.log_scale_weight * scale).mean()
    scale = (_rms_per_orbit(z0) - _rms_per_orbit(x0.detach())).square()
    return (phase + config.temporal_scale_weight * scale).mean()


def factor_scale_loss(z0, x0, K=4, epsilon=1e-6):
    """Mean per-factor squared log-RMS mismatch to detached physical windows."""
    factors = z0.reshape(z0.shape[0], K, z0.shape[-1] // K)
    rms = torch.linalg.vector_norm(factors, dim=-1) / math.sqrt(factors.shape[-1])
    target = _rms_per_orbit(x0.detach())[:, None]
    return (torch.log(rms + epsilon) - torch.log(target + epsilon)).square().mean()


def frequency_coherence_loss(rates):
    """Population variance of log rates across all planes, then mean batch."""
    return torch.log(rates + 1e-8).var(dim=-1, correction=0).mean()


def relative_prediction_loss(prediction: torch.Tensor, target: torch.Tensor,
                             epsilon: float = 1e-4) -> torch.Tensor:
    """Per-orbit mean squared L2 error / detached target energy, then mean B.

    This is exactly the Task 3.1 direct-prediction formula. The state and
    increment AR losses are deliberately left in their archived Task 3 units.
    """
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prediction.ndim != 3 or prediction.shape != target.shape
            or prediction.shape[0] < 1 or prediction.shape[1] < 1
            or prediction.dtype != target.dtype or prediction.device != target.device):
        raise ValueError("prediction and target must have equal (B,H,d) shape, dtype and device")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    numerator = (prediction - target.detach()).square().sum(dim=-1).mean(dim=1)
    denominator = target.detach().square().sum(dim=-1).mean(dim=1) + epsilon
    return (numerator / denominator).mean()


def training_losses(model: M1_IJEPA1D, windows: torch.Tensor,
                    config: TrainConfig) -> dict[str, torch.Tensor]:
    """Compute the full differentiable objective for (B,H+1,L,C) windows.

    Index 0 is the current context and the remaining indices are successive
    macro-step windows. EMA targets include index 0 for the first velocity
    target. Direct predictions all originate at the same online context.
    """
    if not isinstance(model, M1_IJEPA1D):
        raise TypeError("training requires M1, M2, or M3 (not the frozen M0)")
    encoder = model.context_encoder
    expected = (config.batch_size, config.horizon + 1, encoder.L, encoder.C)
    if not isinstance(windows, torch.Tensor) or windows.shape != expected:
        raise ValueError(f"windows must have shape {expected}")
    z0 = encoder(windows[:, 0])
    with torch.no_grad():
        targets = model.target_encoder(windows.flatten(0, 1)).reshape(
            config.batch_size, config.horizon + 1, encoder.d)
    horizons = torch.arange(1, config.horizon + 1, device=z0.device)
    if config.protocol in ("task3_1", "task3_2"):
        # Exp is evaluated exactly once and the differentiable Q is shared by
        # all 64 direct horizons and the entire AR graph. This is not a stale
        # cache across optimizer steps.
        if not (hasattr(model.predictor, "orthogonal_basis")
                and hasattr(model.predictor, "predict_horizons")):
            raise ValueError("Task 3.1 training requires the orthogonal Koopman predictor")
        basis = model.predictor.orthogonal_basis()
        direct = model.predictor.predict_horizons(z0, horizons, basis=basis)
        direct_loss = relative_prediction_loss(
            direct, targets[:, 1:], config.relative_denominator_epsilon)
        predicted_states = model.rollout_autoregressive(z0, config.ar_steps, basis=basis)
    else:
        direct = model.predictor(
            z0[:, None].expand(-1, config.horizon, -1).reshape(-1, encoder.d),
            horizons[None].expand(config.batch_size, -1).reshape(-1),
        ).reshape(config.batch_size, config.horizon, encoder.d)
        direct_loss = _squared_l2_mean(direct, targets[:, 1:])
        predicted_states = model.rollout_autoregressive(z0, config.ar_steps)
    predicted_previous = torch.cat((z0[:, None], predicted_states[:, :-1]), dim=1)
    ar_targets = targets[:, 1:config.ar_steps + 1]
    target_previous = targets[:, :config.ar_steps]
    ar_function = (relative_prediction_loss if config.protocol == "task3_2"
                   and not isinstance(model, M3_IJEPA_SIGReg) else _squared_l2_mean)
    relative_kwargs = ({"epsilon": config.relative_denominator_epsilon}
                       if ar_function is relative_prediction_loss else {})
    ar_state = ar_function(predicted_states, ar_targets, **relative_kwargs)
    ar_velocity = ar_function(predicted_states - predicted_previous,
                             ar_targets - target_previous, **relative_kwargs)
    ar_loss = ar_state + config.velocity_weight * ar_velocity
    zero = z0.new_zeros(())
    variance, temporal, orthogonality, sparsity, isotropic = zero, zero, zero, zero, zero
    factor_scale, coherence = zero, zero
    if isinstance(model, M3_IJEPA_SIGReg):
        isotropic = sigreg_isotropic_loss(z0)
    else:
        if config.protocol in ("task3_1", "task3_2"):
            next_online = encoder(windows[:, config.temporal_lag])
            temporal = temporal_displacement_loss(
                windows[:, 0], windows[:, config.temporal_lag], z0, next_online, config)
        else:
            next_online = encoder(windows[:, 1])
            variance = batch_variance_loss(z0, next_online - z0, config)
        if isinstance(model, M2_JEPA_OPF):
            orthogonality = opf_orthogonality_loss(
                z0, model.K, scale_invariant=config.protocol == "task3_2",
                reduction="mean" if config.protocol == "task3_2" else "legacy")
            if config.protocol == "task3_2":
                factor_scale = factor_scale_loss(z0, windows[:, 0], model.K,
                                                config.log_scale_epsilon)
            if config.sparsity_weight:
                # Optional concentration loss: all but the most active factor
                # contribute their share of the online increment's energy.
                activity = (next_online - z0).reshape(
                    config.batch_size, model.K, model.d_f).square().mean((0, 2))
                sparsity = 1 - activity.max() / activity.sum().clamp_min(1e-12)
        if config.protocol == "task3_2":
            rates = model.predictor.angular_rates(z0, basis=basis)
            coherence = frequency_coherence_loss(rates)
    total = (direct_loss + ar_loss + config.variance_weight * variance
             + config.temporal_weight * temporal
             + config.orthogonality_weight * orthogonality
             + config.sparsity_weight * sparsity + config.sigreg_weight * isotropic
             + config.factor_scale_weight * factor_scale + config.coherence_weight * coherence)
    return {"total": total, "direct": direct_loss, "ar": ar_loss,
            "ar_state": ar_state, "ar_velocity": ar_velocity,
            "variance": variance, "temporal": temporal, "orthogonality": orthogonality,
            "sparsity": sparsity, "isotropic": isotropic,
            "factor_scale": factor_scale, "coherence": coherence}


def ellip_horizon_grid(config):
    """Sparse long direct horizons plus consecutive AR targets, without padding."""
    direct = (list(range(1,config.horizon+1)) if config.ellip_variant=='short'
              else list(range(8,config.horizon+1,8)))
    if not direct:raise ValueError('Long ellipse horizon must include at least step8')
    offsets=sorted(set([0,*direct,*range(1,config.ar_steps+1),config.temporal_lag]))
    return direct,offsets


def elliptic_training_losses(model,first,second,config,offsets):
    """Two same-orbit starts, with no physical parameters or oracle rates.

    Direct and relative AR objectives average over both starts for invariant
    variants. Short uses only its first start and a within-record lag barrier.
    All 32 planes participate in phase-frequency invariance; no coherence
    across different planes is imposed, allowing distinct harmonic rates.
    """
    encoder=model.context_encoder;direct_horizons,_=ellip_horizon_grid(config)
    lookup={k:i for i,k in enumerate(offsets)}
    basis=model.predictor.orthogonal_basis()
    paired=config.ellip_variant!='short'
    anchors=[first,second] if paired else [first]
    zs=[];direct_losses=[];states=[];velocities=[]
    for windows in anchors:
        z=encoder(windows[:,lookup[0]]);zs.append(z)
        with torch.no_grad():
            target=model.target_encoder(windows.flatten(0,1)).reshape(config.batch_size,len(offsets),encoder.d)
        horizons=torch.tensor(direct_horizons,device=z.device)
        predicted=model.predictor.predict_horizons(z,horizons,basis=basis)
        direct_losses.append(relative_prediction_loss(predicted,target[:,[lookup[k] for k in direct_horizons]],config.relative_denominator_epsilon))
        ar=model.rollout_autoregressive(z,config.ar_steps,basis=basis)
        previous=torch.cat([z[:,None],ar[:,:-1]],dim=1)
        t=target[:,[lookup[k] for k in range(1,config.ar_steps+1)]]
        t_prev=target[:,[lookup[k] for k in range(config.ar_steps)]]
        states.append(relative_prediction_loss(ar,t,config.relative_denominator_epsilon))
        velocities.append(relative_prediction_loss(ar-previous,t-t_prev,config.relative_denominator_epsilon))
    direct=torch.stack(direct_losses).mean();ar_state=torch.stack(states).mean();ar_velocity=torch.stack(velocities).mean()
    zero=direct.new_zeros(());temporal=opf=isotropic=inv=zero
    if paired:
        logs=[torch.log(model.predictor.angular_rates(z,basis=basis)+1e-8) for z in zs]
        inv=(logs[0]-logs[1]).square().sum(-1).mean()
    if isinstance(model,M3_IJEPA_SIGReg):
        isotropic=torch.stack([sigreg_isotropic_loss(z) for z in zs]).mean()
    else:
        if paired:x2=second[:,lookup[0]];z2=zs[1]
        else:x2=first[:,lookup[config.temporal_lag]];z2=encoder(x2)
        x1=first[:,lookup[0]]
        displacement=(.5*_rms_per_orbit(x1.detach()-x2.detach())-_rms_per_orbit(zs[0]-z2)).clamp_min(0)
        scale=(torch.log(_rms_per_orbit(zs[0])+1e-6)-torch.log(_rms_per_orbit(x1.detach())+1e-6)).square()
        temporal=(displacement+.2*scale).mean()
        if isinstance(model,M2_JEPA_OPF):
            opf=torch.stack([opf_orthogonality_loss(z,model.K) for z in zs]).mean()
    ar=ar_state+config.velocity_weight*ar_velocity
    weight=config.frequency_invariance_weight if paired else 0.
    total=direct+ar+temporal+config.orthogonality_weight*opf+config.sigreg_weight*isotropic+weight*inv
    return {'total':total,'direct':direct,'ar':ar,'ar_state':ar_state,'ar_velocity':ar_velocity,
            'temporal':temporal,'orthogonality':opf,'isotropic':isotropic,'frequency_invariance':inv}


def _seed_everything(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampling_generator": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    generator.set_state(state["sampling_generator"].cpu())


def _source_fingerprints():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("train.py", "models.py", "losses.py", "dataset.py")}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_model(model: M1_IJEPA1D, trajectories, config: TrainConfig, seed: int,
                checkpoint_path=None, log_callback: Callable | None = None, *,
                resume_from=None, provenance: Mapping | None = None) -> dict:
    """Train a model in place and return scalar history and run provenance.

    Trajectories are placed on the selected GPU once. Random orbit/start pairs
    and their strided target windows are then sampled directly on that device.
    Checkpoints contain the model (including EMA), optimizer, scheduler, all
    RNG states, full configuration, input/source hashes and caller provenance.
    Resume requires the same protocol, seed, input data and model class; it does
    not silently reset or extend the learning-rate schedule.
    """
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    if not isinstance(model, M1_IJEPA1D):
        raise TypeError("training requires M1, M2, or M3")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; production training must run on a GPU, without CPU fallback")
    _seed_everything(seed)
    model.to(device)
    model.train()
    parameter = next(model.context_encoder.parameters())
    encoder = model.context_encoder
    if encoder.L != config.context_length or model.predictor.H_max < config.horizon:
        raise ValueError("model context length/horizon does not match training configuration")
    if isinstance(trajectories, torch.Tensor):
        if trajectories.is_complex():
            raise ValueError("trajectories must be real")
        data_cpu = trajectories.detach().to(device="cpu", dtype=parameter.dtype).contiguous()
    else:
        if np.iscomplexobj(trajectories):
            raise ValueError("trajectories must be real")
        data_cpu = torch.as_tensor(np.asarray(trajectories), dtype=parameter.dtype).contiguous()
    if (data_cpu.ndim != 3 or data_cpu.shape[0] < 1 or data_cpu.shape[2] < encoder.C
            or not torch.isfinite(data_cpu).all()):
        raise ValueError("trajectories must have finite shape (N,T,C>=encoder.C)")
    data_cpu = data_cpu[:, :, :encoder.C].contiguous()
    starts_per_orbit = data_cpu.shape[1] - config.context_length - config.delta * config.horizon + 1
    if starts_per_orbit < 1:
        raise ValueError("trajectories are too short for the complete macro-step training horizon")
    data_hash = hashlib.sha256(data_cpu.numpy().tobytes()).hexdigest()
    data = data_cpu.to(device)
    # unfold is a view: no enormous all-orbit/all-horizon tensor is allocated.
    windows_view = data.unfold(1, config.context_length, 1).permute(0, 1, 3, 2)
    if config.protocol == 'task4_ellip':
        _,ellip_offsets=ellip_horizon_grid(config)
        horizon_offsets=torch.tensor(ellip_offsets,device=device)*config.delta
    else:
        horizon_offsets = torch.arange(config.horizon + 1, device=device) * config.delta
    generator = torch.Generator(device=device).manual_seed(int(seed))
    trainable = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.minimum_learning_rate)
    metadata = {"seed": int(seed), "model_class": type(model).__name__,
                "model_config": {"L": encoder.L, "C": encoder.C, "d": encoder.d,
                                 "arch_type": getattr(encoder, "arch_type", None),
                                 "mlp_variant": getattr(encoder, "mlp_variant",
                                     getattr(getattr(encoder, "base_encoder", None), "mlp_variant", None)),
                                 "H_max": model.predictor.H_max,
                                 "K": getattr(model, "K", None),
                                 "normalize_embeddings": getattr(model, "normalize_embeddings", False),
                                 "predictor_class": type(model.predictor).__name__,
                                 "predictor_dt": getattr(model.predictor, "dt_pred", None),
                                 "encoder_modules": [(name, type(module).__name__)
                                                     for name, module in encoder.named_modules()],
                                 "predictor_modules": [(name, type(module).__name__)
                                                       for name, module in model.predictor.named_modules()]},
                "objective_protocol": config.protocol,
                "data_sha256": data_hash, "source_sha256": _source_fingerprints(),
                "input_shape": list(data_cpu.shape), "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__), "dtype": str(parameter.dtype),
                "device": str(device), "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "caller": dict(provenance or {})}
    history, completed_steps, elapsed_before = [], 0, 0.0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location=device, weights_only=True)
        if saved.get("format_version") != 1 or saved["config"] != asdict(config):
            raise ValueError("checkpoint format or full training configuration differs")
        for name in ("seed", "model_class", "model_config", "objective_protocol",
                     "data_sha256", "dtype", "source_sha256"):
            if saved["provenance"][name] != metadata[name]:
                raise ValueError(f"checkpoint provenance differs: {name}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        _restore_rng(saved["rng_state"], generator)
        history = saved["history"]
        completed_steps, elapsed_before = saved["step"], saved["elapsed_seconds"]
        if not 0 <= completed_steps <= config.steps:
            raise ValueError("checkpoint step is outside configured training schedule")
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    _synchronize(device)
    started = time.perf_counter()

    def checkpoint(step):
        if path is None:
            return
        _synchronize(device)
        payload = {"format_version": 1, "config": asdict(config), "step": step,
                   "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(),
                   "rng_state": _rng_state(generator), "provenance": metadata,
                   "history": history,
                   "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        temporary.replace(path)

    for index in range(completed_steps, config.steps):
        orbit_indices = torch.randint(len(data), (config.batch_size,), generator=generator, device=device)
        starts = torch.randint(starts_per_orbit, (config.batch_size,), generator=generator, device=device)
        batch = windows_view[orbit_indices[:, None], starts[:, None] + horizon_offsets[None]]
        optimizer.zero_grad(set_to_none=True)
        if config.protocol == 'task4_ellip':
            second_starts=torch.randint(starts_per_orbit,(config.batch_size,),generator=generator,device=device)
            second=windows_view[orbit_indices[:,None],second_starts[:,None]+horizon_offsets[None]]
            losses=elliptic_training_losses(model,batch,second,config,ellip_offsets)
        else:
            losses = training_losses(model, batch, config)
        if not bool(torch.isfinite(torch.stack(tuple(losses.values()))).all()):
            raise FloatingPointError(f"nonfinite training loss at step {index + 1}")
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.gradient_clip, error_if_nonfinite=True)
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        fraction = index / max(1, config.steps - 1)
        momentum = config.ema_start + (config.ema_end - config.ema_start) * fraction
        model.update_target_ema(momentum)
        step = index + 1
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            _synchronize(device)
            record = {"step": step, **{name: float(value.detach()) for name, value in losses.items()},
                      "gradient_norm": float(gradient_norm), "lr": lr_used,
                      "next_lr": optimizer.param_groups[0]["lr"], "ema_momentum": momentum,
                      "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            history.append(record)
            if log_callback is not None:
                log_callback(record)
        if step % config.checkpoint_every == 0 or step == config.steps:
            checkpoint(step)
        completed_steps = step
    _synchronize(device)
    model.eval()
    return {"history": history, "completed_steps": completed_steps,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "device": str(device), "config": asdict(config), "provenance": metadata,
            "checkpoint_path": str(path) if path is not None else None}
