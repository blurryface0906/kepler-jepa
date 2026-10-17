"""Experiment 1: T1 spectra of frozen random encoders, with no predictor rollout.

The six estimators see only PCA scores and dt. True orbital parameters are used
to generate the data and evaluate the regressions, never to correct estimates.
All requested results are retained, including failed/undefined estimates.

Run: python -m experiments.exp1_encoder_tautology
Outputs: reports/exp1_results.json, exp1_report.md, figures/exp1_encoder_tautology.png
Exit 0 means the requested empirical criteria passed, not just that inference ran.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tabulate import tabulate
from threadpoolctl import threadpool_limits
import torch

from src.dataset import build_dataset_circ
from src.models import M0_RandomJEPA
from src.simulator import DT, generate_kepler_analytical
from src.spectral import (evaluate_kepler_regression, freq_s1_discrete_fft,
                          freq_s2_interpolated_fft, freq_s3a_dmd_koopman,
                          freq_s3b_phase_unwrap, project_to_2d_mode)

SEEDS = (42, 43, 44, 45, 46)
ENCODERS = {"M0-Linear": "linear", "M0-MLP": "mlp", "M0-Transformer": "transformer"}
METHODS = ("S1", "S2", "S3a_std", "S3b_std", "S3a_diff", "S3b_diff")
LENGTHS = (256, 4096)
ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def _safe_json(value):
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, value):
    # Replacement is atomic; interrupted runs retain the last complete JSON.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_safe_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes():
    return {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in (
        "src/simulator.py", "src/dataset.py", "src/spectral.py", "src/models.py",
        "src/losses.py", "requirements.txt", "experiments/exp1_encoder_tautology.py")}


def _encoder_hash(model):
    digest = hashlib.sha256()
    for key, tensor in model.context_encoder.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _frequency(mode, method):
    if method == "S1":
        return freq_s1_discrete_fft(mode, DT)
    if method == "S2":
        return freq_s2_interpolated_fft(mode, DT)
    if method.startswith("S3a"):
        return freq_s3a_dmd_koopman(mode, DT, use_differences=method.endswith("diff"))
    return freq_s3b_phase_unwrap(mode, DT, use_differences=method.endswith("diff"))


def _regression(axes, frequencies):
    invalid = int(np.count_nonzero(~np.isfinite(frequencies) | (frequencies <= 0)))
    if invalid:
        return {"slope": float("nan"), "slope_abs_error": float("nan"), "r2": float("nan"),
                "one_minus_r2": float("nan"), "log10_one_minus_r2": float("nan"), "invalid": invalid}
    return {**evaluate_kepler_regression(axes, frequencies), "invalid": 0}


def _record_results(seed, encoder, axes, estimates, errors):
    records = []
    for length in LENGTHS:
        for method in METHODS:
            frequencies = np.asarray(estimates[length][method], dtype=np.float64)
            metrics = _regression(axes, frequencies)
            records.append({"seed": seed, "encoder": encoder, "T_eval": length, "method": method,
                            "semi_major_axes": axes.tolist(), "omega_est": frequencies.tolist(),
                            "omega_true": (axes**-1.5).tolist(),
                            "errors": errors[length][method], **metrics})
    return records


def _summary(runs):
    summary = []
    for encoder in ENCODERS:
        for length in LENGTHS:
            for method in METHODS:
                group = [r for r in runs if r["encoder"] == encoder
                         and r["T_eval"] == length and r["method"] == method]
                metrics = {}
                for key in ("slope", "slope_abs_error", "r2", "one_minus_r2"):
                    values = np.asarray([r[key] for r in group], dtype=np.float64)
                    # Do not silently remove undefined seeds from aggregation.
                    metrics[key] = {"mean": float(values.mean()),
                                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
                summary.append({"encoder": encoder, "T_eval": length, "method": method,
                                "n_seeds": len(group), "invalid_orbits": sum(r["invalid"] for r in group),
                                "metrics": metrics})
    return summary


def _criteria(runs):
    checks = []
    for length in LENGTHS:
        for method in ("S3a_diff", "S3b_diff"):
            group = [r for r in runs if r["encoder"] == "M0-Linear"
                     and r["T_eval"] == length and r["method"] == method]
            passed = len(group) == 5 and all(r["invalid"] == 0 and r["slope_abs_error"] < 1e-10
                                            and r["one_minus_r2"] < 1e-20 for r in group)
            checks.append({"criterion": f"H1 Linear {method}, T={length}", "passed": passed,
                           "max_slope_error": max(r["slope_abs_error"] for r in group),
                           "max_residual_ratio": max(r["one_minus_r2"] for r in group)})
    for encoder in ("M0-MLP", "M0-Transformer"):
        method_passes, method_aggregates = {}, {}
        for method in ("S3b_std", "S3b_diff"):
            group = [r for r in runs if r["encoder"] == encoder and r["T_eval"] == 4096
                     and r["method"] == method]
            errors = np.asarray([r["slope_abs_error"] for r in group])
            ratios = np.asarray([r["one_minus_r2"] for r in group])
            valid = len(group) == 5 and all(r["invalid"] == 0 for r in group)
            valid = valid and bool(np.all(np.isfinite(errors)) and np.all(np.isfinite(ratios))
                                   and np.all(errors >= 0) and np.all(ratios >= 0))
            stats = {"mean_slope_abs_error": float(errors.mean()),
                     "max_slope_abs_error": float(errors.max()),
                     "mean_residual_ratio": float(ratios.mean()),
                     "max_residual_ratio": float(ratios.max())}
            method_passes[method] = bool(valid and stats["mean_slope_abs_error"] < 5e-3
                                        and stats["max_slope_abs_error"] < 5e-3
                                        and stats["mean_residual_ratio"] < 1e-5
                                        and stats["max_residual_ratio"] < 1.5e-5)
            method_aggregates[method] = stats
        checks.append({"criterion": f"H1 {encoder}, T=4096, Task3 mean/worst phase criterion",
                       "passed": any(method_passes.values()), "by_method": method_passes,
                       "aggregates": method_aggregates})
    group = [r for r in runs if r["encoder"] == "M0-Linear" and r["T_eval"] == 4096]
    fft = {r["seed"]: r for r in group if r["method"] == "S1"}
    dmd = {r["seed"]: r for r in group if r["method"] == "S3a_diff"}
    phase = {r["seed"]: r for r in group if r["method"] == "S3b_diff"}
    decades = {method: [_improvement_decades(fft[seed]["one_minus_r2"],
                                            estimator[seed]["one_minus_r2"])
                        if fft[seed]["invalid"] == 0 and estimator[seed]["invalid"] == 0
                        else float("nan") for seed in SEEDS]
               for method, estimator in (("S3a_diff", dmd), ("S3b_diff", phase))}
    quantized = all(1e-5 < r["one_minus_r2"] < 5e-2 for r in fft.values())
    improved = all(value >= 15 for values in decades.values() for value in values)
    checks.append({"criterion": "H2 Linear long FFT quantization and >=15-decade continuous improvement",
                   "passed": len(fft) == 5 and quantized and improved, "improvement_decades": decades})
    return checks


def _improvement_decades(discrete_ratio, continuous_ratio):
    """A missing/invalid score cannot be counted as infinite improvement."""
    if (not np.isfinite(discrete_ratio) or discrete_ratio <= 0
            or not np.isfinite(continuous_ratio) or continuous_ratio < 0):
        return float("nan")
    if continuous_ratio == 0:
        return float("inf")
    return math.log10(discrete_ratio) - math.log10(continuous_ratio)


def _format(moment):
    return f"{moment['mean']:.9g} +/- {moment['std']:.3g}"


def _summary_table(summary, length):
    rows = [[r["encoder"], r["method"], _format(r["metrics"]["slope"]),
             _format(r["metrics"]["slope_abs_error"]), _format(r["metrics"]["r2"]),
             _format(r["metrics"]["one_minus_r2"]), r["invalid_orbits"]]
            for r in summary if r["T_eval"] == length]
    return tabulate(rows, headers=["encoder", "method", "slope mean +/- std", "abs(slope+1.5) mean +/- std",
                                   "R2 mean +/- std", "SSE/SST mean +/- std", "invalid /150"], tablefmt="github")


def _plot(runs, figure_path):
    colors = {"M0-Linear": "#2563A8", "M0-MLP": "#D47721", "M0-Transformer": "#278358"}
    styles = {"S1": (":", "o"), "S2": ("--", "s"), "S3b_diff": ("-", "^")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 6.0), gridspec_kw={"width_ratios": [1.08, 1]})
    for encoder in ENCODERS:
        for method, (style, marker) in styles.items():
            group = [r for r in runs if r["encoder"] == encoder and r["T_eval"] == 4096
                     and r["method"] == method]
            x = np.log(np.asarray(group[0]["semi_major_axes"]))
            y = np.log(np.asarray([r["omega_est"] for r in group]))
            mean, std = y.mean(axis=0), y.std(axis=0, ddof=1)
            left.plot(x, mean, color=colors[encoder], linestyle=style, marker=marker,
                      markevery=4, markersize=3.8, linewidth=1.5, alpha=0.9)
            left.fill_between(x, mean - std, mean + std, color=colors[encoder], alpha=0.09)
    left.plot(x, -1.5 * x, color="#222222", linewidth=1, linestyle="-.", label="Kepler reference")
    left.set(xlabel=r"$\log a$", ylabel=r"$\log \hat\omega$", title="(a) Frozen encoders: frequency vs radius")
    left.grid(alpha=0.18)

    # Side-by-side horizontal mini-histograms: each is five seed outcomes.
    # Their ordinate is log residual, so the requested 1e-8 line is horizontal.
    finite_logs = []
    for r in runs:
        if r["T_eval"] == 4096 and np.isfinite(r["one_minus_r2"]) and r["one_minus_r2"] > 0:
            finite_logs.append(math.log10(r["one_minus_r2"]))
    low = min(-30, int(math.floor(min(finite_logs))) - 1)
    bins = np.arange(low, 1.01, 1.0)
    centers = (bins[1:] + bins[:-1]) / 2
    for method_index, method in enumerate(METHODS):
        for encoder_index, encoder in enumerate(ENCODERS):
            group = [r for r in runs if r["encoder"] == encoder and r["T_eval"] == 4096
                     and r["method"] == method]
            values = [math.log10(r["one_minus_r2"]) for r in group
                      if np.isfinite(r["one_minus_r2"]) and r["one_minus_r2"] > 0]
            counts, _ = np.histogram(values, bins=bins)
            positions = method_index + (encoder_index - 1) * 0.26
            widths = 0.23 * counts / 5
            right.barh(centers, widths, height=0.78, left=positions - widths / 2,
                       color=colors[encoder], alpha=0.85)
    right.axhline(-8, color="#992222", linestyle="--", linewidth=1.2)
    right.text(0.02, -7.6, r"JEPA-Anything reference: $1-R^2=10^{-8}$", color="#992222", fontsize=8.5)
    right.set_xticks(range(len(METHODS)), [m.replace("_", "\n") for m in METHODS])
    right.set(xlim=(-0.5, 5.5), ylim=(bins[0], 0), ylabel=r"$\log_{10}(\mathrm{SSE}/\mathrm{SST})$",
              title="(b) Residual distributions across five seeds")
    right.grid(axis="y", alpha=0.18)
    omitted = len([r for r in runs if r["T_eval"] == 4096]) - len(finite_logs)
    right.text(0.5, 0.02, f"Widths: 1-5 seeds; 1-decade bins; omitted: {omitted}",
               transform=right.transAxes, ha="center", va="bottom", fontsize=8)
    handles = [Line2D([0], [0], color=color, linewidth=3, label=encoder) for encoder, color in colors.items()]
    handles += [Line2D([0], [0], color="#555555", linestyle=style, marker=marker,
                      markersize=4, label=method) for method, (style, marker) in styles.items()]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=6, frameon=False)
    figure.suptitle("Experiment 1: encoder sequence T1, no training, 4096 samples, float64", y=0.99, fontsize=13)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.88, wspace=0.26)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=300, facecolor="white", metadata={"Software": "kepler-jepa Exp1"})
    plt.close(figure)


def _markdown(report):
    sections = ["# Experiment 1: frozen encoder spectra", "",
                "T1 only; no training and no predictor rollout. Five seeds: 42–46. "
                "All encoders are initialized directly in float64 and evaluated with dropout disabled.", "",
                "The unchanged test_id grid supplies 30 distinct radii in "
                f"[{report['grid']['min_a']:.9g}, {report['grid']['max_a']:.9g}]. "
                "All estimates, including undefined/failed cases, remain in the JSON and summary.", "",
                "## Results", ""]
    for length in LENGTHS:
        sections.extend([f"### T_eval = {length}", "", _summary_table(report["summary"], length), ""])
    sections.extend(["## Acceptance", "", tabulate(
        [[c["criterion"], "PASS" if c["passed"] else "FAIL"] for c in report["checks"]],
        headers=["criterion", "status"], tablefmt="github"), "",
        "Overall: " + ("PASS" if report["acceptance_passed"] else "NOT MET"), "",
        "Linear thresholds are checked on every seed. For nonlinear encoders Task 3 specifies both "
        "mean(abs(slope+1.5))<0.005, mean(SSE/SST)<1e-5 and worst-seed bounds "
        "max(abs(slope+1.5))<0.005, max(SSE/SST)<1.5e-5. At least one prescribed phase method "
        "must satisfy all four aggregate conditions; no per-orbit or per-seed estimator selection.", "",
        "### Nonlinear phase estimates by seed, T_eval=4096", "",
        tabulate([[r["encoder"], r["seed"], r["method"], r["slope_abs_error"], r["one_minus_r2"],
                   "PASS" if r["invalid"] == 0 and r["slope_abs_error"] < 5e-3
                   and r["one_minus_r2"] < 1.5e-5 else "FAIL"] for r in report["runs"]
                  if r["T_eval"] == 4096 and r["encoder"] != "M0-Linear"
                  and r["method"] in ("S3b_std", "S3b_diff")],
                 headers=["encoder", "seed", "method", "abs(slope+1.5)", "SSE/SST", "status"],
                 tablefmt="github", floatfmt=".9g"), "",
        "## Interpretation limits", "",
        "The isometric linear encoder preserves a harmonic circle in a two-dimensional subspace. "
        "First differences remove additive bias/finite-window centroid shifts. This establishes a "
        "controlled encoder-only baseline and its estimator precision; it does not establish learned gravity.", "",
        "A nonlinear deterministic map preserves periodicity but need not preserve the fundamental spectral "
        "peak or a circular PCA phase. The nonlinear thresholds above are empirical tests, not identities.", "",
        "A successful H2 control demonstrates that a continuous estimator can produce at least eight nines "
        "without training in this setup. It does not identify the procedure used by JEPA-Anything for its "
        "published number without reproduction of that original pipeline. FFT limits depend on the record "
        "length, sampling and estimators; they are not universal limits on discovering a law.", "",
        "R² is a regression score and can conceal frequency bias. Per-orbit estimates and true frequencies "
        "are provided for separate inspection. Constant FFT outputs have undefined R² (JSON null), and no "
        "failed orbit is excluded from a regression.", "",
        "The plot uses the 4096-sample regime. Panel (a) shows mean log frequencies and ±1 sample standard "
        "deviation across seeds. Panel (b) shows side-by-side horizontal histograms of log residual ratio "
        "for each encoder/method; widths represent 1–5 seeds, so the 10⁻⁸ threshold is a horizontal line. "
        "No finite residual is floored to meet an accuracy criterion.", "",
        "## Reproduction", "", "```sh", "python -m experiments.exp1_encoder_tautology", "```", "",
        "Artifacts: [full stdout](exp1_stdout.txt), [detailed JSON](exp1_results.json), "
        "[figure](figures/exp1_encoder_tautology.png). Source SHA-256 hashes, parameter fingerprints, "
        "environment, numerical settings, timings, failed-estimate messages and criteria are stored in JSON.", ""])
    return "\n".join(sections)


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    report = {"experiment": "exp1_encoder_tautology", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"seeds": list(SEEDS), "encoders": ENCODERS, "T_eval": list(LENGTHS),
                         "L": 16, "C": 4, "d": 64, "T_total": 4112, "dt": DT, "mu": 1,
                         "dtype": "float64 initialized directly", "device": "cpu",
                         "batch_size": args.batch_size, "threads": args.threads,
                         "training_steps": 0, "mode": "T1 encoder sequence",
                         "PCA": "centered unwhitened two-component SVD", "methods": METHODS},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: importlib.metadata.version(name) for name in
                                 ("numpy", "scipy", "scikit-learn", "torch", "matplotlib", "tabulate", "threadpoolctl")}},
              "source_sha256": _source_hashes(), "runs": [], "encoder_runs": []}
    json_path = args.output / "exp1_results.json"
    start = time.perf_counter()
    print("Experiment 1: encoder sequence T1, NO TRAINING, no predictor rollout", flush=True)
    print("Command: python -m experiments.exp1_encoder_tautology" + (" " + " ".join(sys.argv[1:]) if sys.argv[1:] else ""), flush=True)
    print("Configuration: " + json.dumps(report["config"]), flush=True)
    print("Environment: " + json.dumps(report["environment"]), flush=True)
    for seed in SEEDS:
        batch = build_dataset_circ("test_id", seed)
        selected = (batch["meta"]["a"] >= 0.5) & (batch["meta"]["a"] <= 2)
        axes, initial_phase = batch["meta"]["a"][selected], batch["meta"]["M0"][selected]
        if len(axes) != 30:
            raise ValueError("The specified test_id grid must supply exactly 30 radii in [.5,2]")
        trajectories = generate_kepler_analytical(axes[:, None], np.zeros((30, 1)),
                                                  np.arange(4112, dtype=np.float64)[None] * DT,
                                                  M0=initial_phase[:, None])
        if "grid" not in report:
            report["grid"] = {"source": "build_dataset_circ('test_id'), unchanged grid filtered to [.5,2]",
                              "n_orbits": 30, "min_a": float(axes.min()), "max_a": float(axes.max()),
                              "a": axes.tolist(), "min_long_periods": float(4095 * DT * axes.max()**-1.5 / (2 * np.pi)),
                              "max_long_periods": float(4095 * DT * axes.min()**-1.5 / (2 * np.pi))}
        for encoder, architecture in ENCODERS.items():
            torch.manual_seed(seed)
            model = M0_RandomJEPA(L=16, C=4, d=64, H_max=128,
                                  arch_type=architecture, dtype=torch.float64,
                                  mlp_variant="legacy_gelu").eval()
            before = _encoder_hash(model)
            model_start = time.perf_counter()
            estimates = {length: {method: [] for method in METHODS} for length in LENGTHS}
            errors = {length: {method: [] for method in METHODS} for length in LENGTHS}
            print(f"seed={seed} encoder={encoder}: starting 30 orbits", flush=True)
            with torch.inference_mode():
                for orbit, trajectory in enumerate(trajectories):
                    windows = np.lib.stride_tricks.sliding_window_view(trajectory, 16, axis=0).transpose(0, 2, 1)[:4096]
                    windows = torch.from_numpy(np.array(windows, dtype=np.float64, copy=True))
                    latent = model.encode_sequence(windows, batch_size=args.batch_size).numpy()
                    for length in LENGTHS:
                        mode = project_to_2d_mode(latent[:length])
                        for method in METHODS:
                            try:
                                frequency = _frequency(mode, method)
                                error = None
                            except ValueError as exc:
                                frequency, error = float("nan"), str(exc)
                            estimates[length][method].append(frequency)
                            errors[length][method].append(error)
                    if (orbit + 1) % 10 == 0:
                        print(f"seed={seed} encoder={encoder}: {orbit + 1}/30 orbits, "
                              f"{time.perf_counter() - model_start:.2f}s", flush=True)
            after = _encoder_hash(model)
            if before != after or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("The random baseline changed or contains trainable parameters")
            report["runs"].extend(_record_results(seed, encoder, axes, estimates, errors))
            report["encoder_runs"].append({"seed": seed, "encoder": encoder,
                                           "encoder_sha256_before": before, "encoder_sha256_after": after,
                                           "trainable_parameters": 0, "training_steps": 0,
                                           "elapsed_seconds": time.perf_counter() - model_start})
            _write_json(json_path, report)
            print(f"seed={seed} encoder={encoder}: complete, weights unchanged", flush=True)
    if report["source_sha256"] != _source_hashes():
        raise RuntimeError("Source changed while the experiment was running; results are not a single source revision")
    report["summary"] = _summary(report["runs"])
    report["checks"] = _criteria(report["runs"])
    report["acceptance_passed"] = all(check["passed"] for check in report["checks"])
    report["elapsed_seconds"] = time.perf_counter() - start
    report["status"] = "completed"
    _plot(report["runs"], args.output / "figures/exp1_encoder_tautology.png")
    _write_json(json_path, report)
    (args.output / "exp1_report.md").write_text(_markdown(report), encoding="utf-8")
    for length in LENGTHS:
        print(f"\nT_eval={length}: mean +/- sample std across five seeds; no undefined seed removed")
        print(_summary_table(report["summary"], length))
    print("\nACCEPTANCE")
    print(tabulate([[c["criterion"], "PASS" if c["passed"] else "FAIL"] for c in report["checks"]],
                   headers=["criterion", "status"], tablefmt="github"))
    print(f"Elapsed seconds: {report['elapsed_seconds']:.2f}")
    print("Overall: " + ("PASS (exit 0)" if report["acceptance_passed"] else "NOT MET (exit 1)"))
    return 0 if report["acceptance_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/Task_3")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "exp1_stdout.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), threadpool_limits(limits=args.threads):
            return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
