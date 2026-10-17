"""Task 3.2: OPF remediation, frozen Koopman and out-of-sample controls.

Production defaults never fall back to CPU or shorten the requested protocol.
For engineering checks only, --smoke-test writes under Task_3_2/smoke and labels
every artifact accordingly. --preflight checks the GPU without training.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.dataset import build_dataset_circ
from src.evaluation import (METHODS, evaluate_frequencies, evaluate_regimes,
                            evaluate_scale_preservation, evaluate_oos_scale_probes, regime_metadata)
from src.models import M0_RandomJEPA, M0_Koopman, M1_IJEPA1D, M2_JEPA_OPF, M3_IJEPA_SIGReg
from src.simulator import DT, generate_kepler_analytical
from src.spectral import project_to_2d_mode
from src.train import TrainConfig, train_model
from .exp1_encoder_tautology import _safe_json, _Tee, _write_json
from .exp2_reporting import write_report

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_METHOD = "S3b_diff"
MODELS = {"M0": M0_RandomJEPA, "M0_Koopman": M0_Koopman,
          "M1": M1_IJEPA1D, "M2": M2_JEPA_OPF, "M3": M3_IJEPA_SIGReg}


def gpu_preflight():
    available = torch.cuda.is_available()
    gpus = []
    if available:
        for index in range(torch.cuda.device_count()):
            info = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": info.name, "total_memory_bytes": info.total_memory,
                         "capability": list(torch.cuda.get_device_capability(index))})
    return {"cuda_available": available, "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
            "platform": platform.platform(), "gpus": gpus,
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def _source_hashes():
    files = ("src/models.py", "src/losses.py", "src/train.py", "src/evaluation.py", "src/dataset.py",
             "src/simulator.py", "src/spectral.py", "experiments/exp2_circular_trained.py",
             "experiments/exp2_reporting.py", "requirements-gpu.txt")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def _weights_hash(model, original_dtype=None):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu()
        if original_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(original_dtype)
        digest.update(key.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model(name, seed, device):
    torch.manual_seed(seed)
    kwargs = {"L": 16, "C": 4, "d": 64, "H_max": 512 if name == "M0" else 64,
              "arch_type": "mlp", "mlp_variant": "odd_residual", "dtype": torch.float32}
    if name not in ("M0", "M0_Koopman"):
        kwargs.update(predictor_type="koopman", dt_pred=DT * 4)
    if name == "M3":
        kwargs["normalize_embeddings"] = True
    return MODELS[name](**kwargs).to(device)


def _test_batches(seed, raw_count):
    batches = {}
    for split, source in (("test_id_sub", "test_id"), ("test_ood", "test_ood")):
        meta = build_dataset_circ(source, seed)["meta"]
        selected = (meta["a"] >= .5) & (meta["a"] <= 2) if split == "test_id_sub" else np.ones(len(meta["a"]), bool)
        a, phase = meta["a"][selected], meta["M0"][selected]
        states = generate_kepler_analytical(a[:,None], np.zeros((len(a),1)),
                                            np.arange(raw_count, dtype=np.float64)[None] * DT,
                                            M0=phase[:,None])
        batches[split] = {"trajectories": states, "a": a}
    return batches


def _criterion(checks, label, records, predicate, *, expected=5):
    identities = {(row["seed"], row.get("regime")) for row in records}
    valid = (len(records) == expected and len(identities) == expected
             and {row["seed"] for row in records} == set(SEEDS))
    passed = valid and all(predicate(row) for row in records)
    checks.append({"criterion": label, "passed": bool(passed),
                   "n_seed_records": len(records), "expected_seed_records": expected,
                   "details": [{"seed": row["seed"], **{key: row.get(key) for key in
                       ("regime", "slope_abs_error", "r2", "invalid", "state_norm_r2",
                        "phase_norm_r2", "axis_probe_r2")}}
                               for row in records]})


def acceptance_checks(report):
    """Sixteen fixed ID checks; T4 must pass at both 64 and 512 horizons.

    Neither the leading factor nor a different estimator can replace the
    predeclared full-64D S3b_diff result. OOD remains fully reported diagnostics.
    """
    checks = []
    objective = report["protocol"].get("training", {}).get("protocol", "task3_1")
    production = not report["protocol"]["smoke_test"]
    expected_runs = {(seed, model) for seed in SEEDS for model in ("M1", "M2", "M3")}
    complete = (len(report["training_runs"]) == 15
                and {(r["seed"], r["model"]) for r in report["training_runs"]} == expected_runs and all(
        row.get("completed_steps") == 2000 and row.get("device", "").startswith("cuda")
        and row.get("status") == "completed" and row.get("config", {}).get("batch_size") == 256
        and row.get("config", {}).get("protocol") == objective for row in report["training_runs"]))
    checks.append({"criterion": "All 15 CUDA training runs completed 2000 steps, batch 256",
                   "passed": bool(production and complete
                                  and report["protocol"]["primary_method"] == PRIMARY_METHOD)})
    method = PRIMARY_METHOD
    frequencies = [r for r in report["frequency_runs"] if r["split"] == "test_id_sub" and r["method"] == method]
    for model in ("M0", "M1", "M2"):
        selected = [r for r in frequencies if r["model"] == model and r["regime"] == "T1"]
        _criterion(checks, f"ID T1 {model}: slope error<.01 and R2>.9999 ({method}, full64)", selected,
                   lambda r: r["invalid"] == 0 and _finite(r["slope_abs_error"])
                   and r["slope_abs_error"] < .01 and _finite(r["r2"]) and r["r2"] > .9999)
    for regime in ("T3", "T4"):
        regimes = ("T4", "T4_512") if regime == "T4" else ("T3",)
        expected = 10 if regime == "T4" else 5
        selected = [r for r in frequencies if r["model"] == "M0" and r["regime"] in regimes]
        suffix = "64 AND 512" if regime == "T4" else "512"
        _criterion(checks, f"ID {regime} M0 fails law recovery ({method}, {suffix}, full64)", selected,
                   lambda r: r["invalid"] > 0 or not _finite(r["r2"])
                   or (_finite(r["slope_abs_error"]) and r["slope_abs_error"] > .3)
                   or r["r2"] < .80, expected=expected)
        for model in ("M1", "M2"):
            selected = [r for r in frequencies if r["model"] == model and r["regime"] in regimes]
            _criterion(checks, f"ID {regime} {model}: slope error<.05 and R2>.99 ({method}, {suffix}, full64)", selected,
                       lambda r: r["invalid"] == 0 and _finite(r["slope_abs_error"])
                       and r["slope_abs_error"] < .05 and _finite(r["r2"]) and r["r2"] > .99,
                       expected=expected)
    for regime in ("T1", "T3"):
        for model in ("M1", "M2", "M3"):
            selected = [r for r in report["scale_runs"] if r["split"] == "test_id_sub"
                        and r["model"] == model and r["regime"] == regime]
            _criterion(checks, f"H4 ID {regime} {model}: " + ("state and phase R2<.05" if model == "M3"
                       else "phase R2>.90 AND axis-probe R2>.95"),
                       selected, lambda r, model=model: r.get("status") == "completed"
                       and _finite(r.get("phase_norm_r2"))
                       and ((_finite(r.get("state_norm_r2")) and r["state_norm_r2"] < .05
                             and r["phase_norm_r2"] < .05) if model == "M3" else
                            (_finite(r.get("axis_probe_r2")) and r["phase_norm_r2"] > .90
                             and r["axis_probe_r2"] > .95)))
    if objective == "task3_2":
        for regime, regimes in (("T3", ("T3",)), ("T4", ("T4", "T4_512"))):
            selected = [r for r in frequencies if r["model"] == "M0_Koopman" and r["regime"] in regimes]
            _criterion(checks, f"Control ID {regime} M0_Koopman fails law recovery ({method}, full64)",
                       selected, lambda r: r["invalid"] > 0 or not _finite(r["r2"])
                       or (_finite(r["slope_abs_error"]) and r["slope_abs_error"] > .3)
                       or r["r2"] < .80, expected=10 if regime == "T4" else 5)
    return checks


def _finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(np.isfinite(value))


def _active_factor_sequences(latent, K=4):
    """Choose each orbit's factor by temporal variance only, before frequencies.

    A different factor may be active on a different orbit. No estimated slope,
    true radius, frequency error, acceptance threshold, or seed enters selection.
    """
    values = np.asarray(latent, dtype=np.float64)
    factors = values.reshape(values.shape[0], values.shape[1], K, values.shape[2] // K)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = factors.var(axis=1, ddof=1).sum(axis=-1)
    indices = np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    chosen = np.stack([factors[i, :, index, :] for i, index in enumerate(indices)])
    # A failed full orbit must not become an apparently valid factor diagnostic
    # merely because another factor happened to remain finite.
    chosen[~np.isfinite(values).all(axis=(1, 2))] = np.nan
    return chosen, indices.tolist(), scores.tolist()


def _predictor_diagnostics(model, batch, sequences, config):
    """Measure actual iterative AR, including finite-precision operator error."""
    trajectories = batch["trajectories"]
    parameter = next(model.parameters())
    with torch.inference_mode():
        z0 = model.encoder(torch.as_tensor(trajectories[:, :16].copy(),
                                          dtype=parameter.dtype, device=parameter.device))
        initial = z0.cpu().numpy()
        ar = sequences["T3"]
        direct = sequences[f"T4_{ar.shape[1]}"]
        initial_norm = np.linalg.norm(initial, axis=-1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            drift = np.abs(np.linalg.norm(ar, axis=-1) / initial_norm[:, None] - 1)
            agreement = np.linalg.norm(ar - direct, axis=-1) / initial_norm[:, None]
        result = {"norm_relative_drift_max": float(np.max(drift)),
                  "ar_direct_relative_error_max": float(np.max(agreement)),
                  "action_log_drift_max": None, "orthogonality_max_abs": None,
                  "structured_predictor": hasattr(model.predictor, "orthogonal_basis")}
        if result["structured_predictor"]:
            Q = model.predictor.orthogonal_basis()
            actions0 = model.predictor.plane_log_actions(z0, basis=Q)
            tensor_ar = torch.as_tensor(ar, dtype=parameter.dtype, device=parameter.device)
            actions = model.predictor.plane_log_actions(tensor_ar.reshape(-1, ar.shape[-1]), basis=Q)
            actions = actions.reshape(ar.shape[0], ar.shape[1], -1)
            result["action_log_drift_max"] = float((actions - actions0[:, None]).abs().max().cpu())
            result["orthogonality_max_abs"] = float((Q.T @ Q - torch.eye(Q.shape[0],
                                                         dtype=Q.dtype, device=Q.device)).abs().max().cpu())
            rates = model.predictor.angular_rates(z0, basis=Q)
            dispersion = torch.log(rates + 1e-8).var(dim=-1, correction=0)
            result.update(initial_plane_rates=rates.cpu().tolist(),
                          log_rate_variance_mean=float(dispersion.mean().cpu()),
                          log_rate_variance_max=float(dispersion.max().cpu()))
    count, lag = sequences["T1"].shape[1], config.temporal_lag
    starts = config.delta * np.arange(1, count + 1)
    windows = trajectories[:, starts[:, None] + np.arange(16)[None, :], :]
    true_latent = sequences["T1"]
    input_rms = np.sqrt(np.mean((windows[:, lag:] - windows[:, :-lag]) ** 2, axis=(1, 2, 3)))
    latent_rms = np.sqrt(np.mean((true_latent[:, lag:] - true_latent[:, :-lag]) ** 2, axis=(1, 2)))
    result.update(temporal_lag=lag, input_temporal_rms=input_rms.tolist(),
                  latent_temporal_rms=latent_rms.tolist(),
                  input_temporal_rms_median=float(np.median(input_rms)),
                  latent_temporal_rms_median=float(np.median(latent_rms)),
                  temporal_displacement_ratio_median=float(np.median(latent_rms / np.maximum(input_rms, 1e-12))))
    return result


def _absolute_frequency_statistics(result):
    """Keep all orbits: one invalid frequency makes aggregate q undefined."""
    a = np.asarray(result["semi_major_axes"], dtype=np.float64)
    omega = np.asarray([np.nan if x is None else x for x in result["omega_est"]])
    q = omega / a ** -1.5
    valid = bool(np.isfinite(q).all() and (q > 0).all())
    return {"q": q.tolist(), "q_median": float(np.median(q)) if valid else float("nan"),
            "q_min": float(q.min()) if valid else float("nan"),
            "q_max": float(q.max()) if valid else float("nan"),
            "max_abs_q_error": float(np.abs(q - 1).max()) if valid else float("nan"),
            "q_reference": "mu=1 physical omega=a**(-1.5); diagnostic only"}


def _probe_training_batch(seed, raw_count, smoke=False):
    meta = build_dataset_circ("train", seed)["meta"]
    a, phase = meta["a"], meta["M0"]
    if smoke:
        a, phase = a[::12], phase[::12]
    states = generate_kepler_analytical(a[:, None], np.zeros((len(a), 1)),
                                       np.arange(raw_count)[None] * DT, M0=phase[:, None])
    return {"trajectories": states, "a": a}


def _evaluate(model, name, seed, batches, report, config, n_eval, probe_batch=None):
    # The checkpoint contains the learned float32 values. Promotion changes
    # arithmetic precision, not their information content or learned weights.
    # This explicit stage isolates long-AR roundoff from mathematical collapse.
    model.to(dtype=torch.float64).eval()
    fitted = {}
    if probe_batch is not None:
        train_sequences = evaluate_regimes(model, probe_batch["trajectories"], N_eval=n_eval, L=16,
                                           delta=config.delta, H_train=config.horizon, dt=DT,
                                           device=config.device, H_extended=n_eval)
        for regime in ("T1", "T3"):
            try:
                fitted[regime] = evaluate_scale_preservation(train_sequences[regime], probe_batch["a"],
                                                             dt_pred=config.delta * DT)
                report["probe_training_runs"].append({"seed": seed, "model": name, "split": "train",
                                                       "regime": regime, "status": "completed", **fitted[regime]})
            except ValueError as error:
                report["probe_training_runs"].append({"seed": seed, "model": name, "split": "train",
                                                       "regime": regime, "status": "invalid", "error": str(error)})
        del train_sequences
    evaluations = []
    for base_split, base_batch in batches.items():
        base_sequences = evaluate_regimes(model, base_batch["trajectories"], N_eval=n_eval, L=16,
                                          delta=config.delta, H_train=config.horizon, dt=DT,
                                          device=config.device, H_extended=n_eval)
        evaluations.append((base_split, base_batch, base_sequences))
        if base_split == "test_ood" and config.protocol == "task3_2":
            for split, mask in (("test_ood_inner", base_batch["a"] < .5),
                                 ("test_ood_outer", base_batch["a"] > 5)):
                evaluations.append((split, {"a": base_batch["a"][mask],
                                             "trajectories": base_batch["trajectories"][mask]},
                                     {k: v[mask] for k, v in base_sequences.items()}))
    for split, batch, sequences in evaluations:
        a = batch["a"]
        report.setdefault("test_grids", {})[split] = a.tolist()
        for regime, latent in sequences.items():
            results = evaluate_frequencies(latent, a, dt=config.delta * DT)
            for method, result in results.items():
                report["frequency_runs"].append({"seed": seed, "model": name, "split": split,
                                                  "regime": regime, **result,
                                                  **_absolute_frequency_statistics(result)})
            if name == "M2":
                active, indices, scores = _active_factor_sequences(latent, model.K)
                for method, result in evaluate_frequencies(active, a, dt=config.delta * DT).items():
                    report["factor_frequency_runs"].append({"seed": seed, "model": name, "split": split,
                        "regime": regime, "active_factor_indices": indices,
                        "factor_temporal_variance_scores": scores,
                        "factor_selection": "per-orbit maximum temporal variance; first index breaks ties",
                        **result})
            if regime in ("T1", "T3"):
                try:
                    scale = {"status": "completed", **evaluate_scale_preservation(latent, a,
                                                                                   dt_pred=config.delta * DT)}
                except ValueError as error:
                    scale = {"status": "invalid", "error": str(error),
                             "scale_r2_cycle_radius": float("nan"), "scale_r2_state_norm": float("nan"),
                             "energy_r2": float("nan"), "state_norm_r2": float("nan"),
                             "phase_norm_r2": float("nan"), "axis_probe_r2": float("nan")}
                report["scale_runs"].append({"seed": seed, "model": name, "split": split,
                                              "regime": regime, **scale})
                if probe_batch is not None:
                    oos = {"status": "invalid", "error": "Train or test scale evaluation invalid",
                           "oos_phase_norm_r2": float("nan"), "oos_axis_probe_r2": float("nan")}
                    if regime in fitted and scale["status"] == "completed":
                        oos = evaluate_oos_scale_probes(fitted[regime], scale)
                    report["oos_scale_runs"].append({"seed": seed, "model": name,
                                                    "split": split, "regime": regime, **oos})
            if seed == 42 and split == "test_id_sub" and regime == "T3" and name in ("M0", "M0_Koopman", "M1", "M2"):
                index = int(np.argmin(np.abs(np.log(a))))
                try:
                    portrait = {"U": project_to_2d_mode(latent[index]).tolist(), "error": None}
                except (ValueError, np.linalg.LinAlgError) as error:
                    portrait = {"U": None, "error": str(error)}
                report["portraits"].append({"seed": seed, "model": name, "split": split,
                                             "orbit_index": index, "a": float(a[index]), **portrait})
        report["predictor_diagnostics"].append({"seed": seed, "model": name, "split": split,
                                                **_predictor_diagnostics(model, batch, sequences, config)})
        print(f"seed={seed} model={name} split={split}: evaluated T1/T2/T3={n_eval}, "
              f"T4={config.horizon}, T4_{n_eval}={n_eval}, dtype=float64", flush=True)


def run(args):
    config = (TrainConfig(steps=3, batch_size=8, horizon=4, ar_steps=2, temporal_lag=2,
                           device="cpu", log_every=1, checkpoint_every=1, smoke_test=True,
                           protocol="task3_2")
              if args.smoke_test else TrainConfig(device=args.device, protocol="task3_2"))
    preflight = gpu_preflight()
    if not args.smoke_test and not preflight["cuda_available"]:
        _write_json(args.output / "gpu_preflight.json", preflight)
        print("CUDA unavailable. Full training was not started; no CPU substitution.", flush=True)
        return 2
    seeds, n_eval = ((42,), 16) if args.smoke_test else (SEEDS, 512)
    metadata = regime_metadata(N_eval=n_eval, delta=config.delta, H_train=config.horizon, H_extended=n_eval)
    raw_count = metadata["required_raw_samples"] if args.smoke_test else 2100
    report = {"experiment": "exp2_circular_trained", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"smoke_test": args.smoke_test, "seeds": list(seeds),
                           "training": asdict(config), "arch_type": "mlp", "mlp_variant": "odd_residual",
                           "models": list(MODELS),
                           "splits": ["test_id_sub", "test_ood", "test_ood_inner", "test_ood_outer"],
                           "d": 64, "C": 4, "raw_test_samples": raw_count,
                           "primary_method": PRIMARY_METHOD, "frequency_methods": list(METHODS),
                           "spectral_input": "primary full d=64 latent, per-orbit unwhitened centered PCA",
                           "factor_evaluation": "M2 per-orbit max temporal variance; additional diagnostic, never substitutes primary",
                           "predictor": "M1/M2/M3 action-angle Koopman; Q=exp(S-S.T), M2 Q four16Dblocks, learned softplus rate_net",
                           "M0_predictor": "frozen random unstructured MLP, H_max=512; no Kepler metadata",
                           "training_dtype": "float32", "evaluation_dtype": "float64",
                           "evaluation_precision": "promote learned float32 parameters after checkpoint; no further training or float64 initialization",
                           "M3_normalization": "encoder/target pointwise feature-centered sphere radius sqrt(d); generated Koopman states retain norm without re-centering",
                           "M3_objective": "relative direct+legacy AR+beta*SIGReg, beta=1; no M1/M2 temporal barrier",
                           "M1_M2_objective": "relative direct+relative AR state/increment+log temporal1.0+coherence.1; M2 scale-invariant mean OPF.2+factor log-scale.25",
                           "OPF": "per-state RMS sqrt(mean(z²)+1e-6); population batch standardization; mean of 3072 ordered off-block correlation squares",
                           "frequency_coherence": "population variance of log(omega+1e-8) across all32 planes; weight0.1; M1/M2 only",
                           "M0_Koopman": "frozen encoder/predictor identically initialized to M1 at same seed; no optimizer",
                           "oos_probes": "fit all200 train circular orbits separately for T1/T3, freeze coefficients and constant-column selection; score all test orbits without refit",
                           "acceptance_accounting": "16 inherited criteria plus2 M0_Koopman controls; formal section3 failure threshold .3/R2 .8, not narrative >1.0",
                           "phase_norm_probe": "response log latent RMS vs feature 0.5*log(a^2+1/a); constant response zero-score convention disclosed",
                           "sparsity_weight": 0.0, "regimes": metadata,
                           "acceptance_split": "test_id_sub", "ood": "diagnostic; no OOD threshold specified"},
              "environment": {**preflight, **{package: importlib.metadata.version(package) for package in
                                               ("numpy", "scipy", "scikit-learn", "matplotlib", "tabulate", "threadpoolctl")}},
              "source_sha256": _source_hashes(), "training_runs": [], "frequency_runs": [],
              "scale_runs": [], "factor_frequency_runs": [], "predictor_diagnostics": [],
              "probe_training_runs": [], "oos_scale_runs": [], "control_weight_checks": [],
              "portraits": [], "checks": []}
    json_path = args.output / "exp2_circular_results.json"
    _write_json(json_path, report)
    print("Task 3.2: " + ("CPU ENGINEERING SMOKE ONLY" if args.smoke_test else "FULL CUDA EXPERIMENT"), flush=True)
    print("Protocol: " + json.dumps(report["protocol"]), flush=True)
    print("GPU: " + json.dumps(preflight), flush=True)
    started = time.perf_counter()
    for seed in seeds:
        batches = _test_batches(seed, raw_count)
        report.setdefault("test_grids", {split: batch["a"].tolist() for split, batch in batches.items()})
        training_data = build_dataset_circ("train", seed)["trajectories"]
        probe_batch = _probe_training_batch(seed, raw_count, args.smoke_test)
        for name in MODELS:
            model = _model(name, seed, config.device)
            frozen_hash = _weights_hash(model) if name in ("M0", "M0_Koopman") else None
            if name not in ("M0", "M0_Koopman"):
                checkpoint = args.output / "checkpoints" / f"{name}_seed{seed}.pt"
                resume_path = checkpoint if args.resume and checkpoint.exists() else None
                print(f"seed={seed} model={name}: training steps={config.steps}, batch={config.batch_size}, device={config.device}", flush=True)

                def log(record):
                    print(f"seed={seed} model={name} step={record['step']}/{config.steps} "
                          f"loss={record['total']:.7g} direct={record['direct']:.6g} "
                          f"ar={record['ar']:.6g} temporal={record.get('temporal', 0):.6g} "
                          f"ortho={record['orthogonality']:.6g} iso={record['isotropic']:.6g} "
                          f"factor_scale={record.get('factor_scale', 0):.6g} coh={record.get('coherence', 0):.6g} "
                          f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)
                    report["progress"] = {"seed": seed, "model": name, **record}
                    _write_json(json_path, report)

                try:
                    training = train_model(model, training_data, config, seed,
                                           checkpoint_path=checkpoint, log_callback=log,
                                           resume_from=resume_path, provenance={"experiment": "exp2_task3_2", "seed": seed,
                                                                               "source_sha256": report["source_sha256"]})
                    report["training_runs"].append({"seed": seed, "model": name, "status": "completed", **training})
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    report["training_runs"].append({"seed": seed, "model": name, "status": "failed",
                                                    "error": str(error), "completed_steps": None,
                                                    "device": config.device})
                    print(f"seed={seed} model={name}: TRAINING FAILED: {error}", flush=True)
                    _write_json(json_path, report)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            _evaluate(model, name, seed, batches, report, config, n_eval, probe_batch)
            if frozen_hash is not None:
                after = _weights_hash(model, original_dtype=torch.float32)
                report["control_weight_checks"].append({"seed": seed, "model": name,
                    "before_sha256": frozen_hash, "after_sha256": after, "unchanged": frozen_hash == after,
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
                if frozen_hash != after:
                    raise RuntimeError("Frozen control weights changed during evaluation")
            _write_json(json_path, report)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if report["source_sha256"] != _source_hashes():
        raise RuntimeError("Source changed during the experiment; refusing mixed-source results")
    report["checks"] = acceptance_checks(report)
    report["acceptance_passed"] = all(row["passed"] for row in report["checks"])
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "smoke_completed" if args.smoke_test else "completed"
    report["execution_completed"] = len(report["training_runs"]) == len(seeds) * 3 and all(
        row["status"] == "completed" and row["completed_steps"] == config.steps for row in report["training_runs"])
    _write_json(json_path, report)
    write_report(report, args.output)
    for check in report["checks"]:
        print(("PASS" if check["passed"] else "FAIL") + ": " + check["criterion"], flush=True)
    print("Execution completed: " + str(report["execution_completed"]), flush=True)
    if args.smoke_test:
        print("SMOKE ONLY: no scientific acceptance or full GPU training claimed.", flush=True)
        return 0 if report["execution_completed"] else 1
    print("Overall scientific acceptance: " + ("PASS (exit 0)" if report["acceptance_passed"] else "NOT MET (exit 1)"), flush=True)
    return 0 if report["acceptance_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/Task_3_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--primary-method", choices=(PRIMARY_METHOD,), default=PRIMARY_METHOD)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.smoke_test:
        args.output = args.output / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.preflight:
        preflight = gpu_preflight()
        _write_json(args.output / "gpu_preflight.json", preflight)
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["cuda_available"] else 2
    log_name = "exp2_circular_stdout.txt"
    with (args.output / log_name).open("w", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), threadpool_limits(limits=args.threads):
            return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
