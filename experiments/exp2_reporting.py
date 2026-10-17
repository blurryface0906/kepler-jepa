"""Task 3.2 reports: retain undefined outcomes and separate execution/criteria.

The publication figure is created only when its five-seed input records exist.
An unavailable GPU or incomplete run yields a pending Markdown report, never a
placeholder presented as a final experimental figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tabulate import tabulate

MODELS = ("M0", "M1", "M2", "M3")
REGIMES = ("T1", "T2", "T3", "T4", "T4_512")
METHODS = ("S2", "S3a_diff", "S3b_diff", "S3b_std")
SPLITS = ("test_id_sub", "test_ood")
COLORS = {"M0": "#667085", "M1": "#2468A2", "M2": "#D27A28", "M3": "#8B4E9F"}
COLORS["M0_Koopman"] = "#32845B"


def _models(report):
    return tuple(report.get("protocol", {}).get("models", MODELS))


def _splits(report):
    return tuple(report.get("protocol", {}).get("splits", SPLITS))


def _comparison_models(report):
    return tuple(name for name in _models(report) if name != "M3")


def _seeds(report):
    protocol = report.get("protocol", {})
    return tuple(protocol.get("seeds", [42, 43, 44, 45, 46])) if isinstance(protocol, dict) else (42, 43, 44, 45, 46)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_values(records, key):
    return np.asarray([_number(record.get(key)) for record in records], dtype=np.float64)


def _moment(records, key):
    """An undefined seed makes the aggregate undefined; no nanmean selection."""
    values = _finite_values(records, key)
    invalid = int(np.count_nonzero(~np.isfinite(values)))
    if not len(values):
        return "pending"
    if invalid:
        return f"undefined ({invalid}/{len(values)} seeds)"
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return f"{values.mean():.6g} ± {std:.3g}"


def _group(records, **selection):
    return [record for record in records if all(record.get(key) == value for key, value in selection.items())]


def _plain_json(value):
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    return value


def _figure_missing(report):
    expected = set(_seeds(report))
    missing = []
    if report.get("protocol", {}).get("smoke_test", False):
        missing.append("Engineering smoke test only; the publication figure requires the full five-seed CUDA experiment.")
    production_seeds = {42, 43, 44, 45, 46}
    if expected != production_seeds:
        missing.append("Publication figure requires the five prespecified seeds 42–46.")
    training = report.get("training_runs", [])
    completed = [record for record in training
                 if record.get("status") == "completed"
                 and record.get("completed_steps") == 2000
                 and str(record.get("device", "")).startswith("cuda")
                 and record.get("config", {}).get("batch_size",
                         report.get("protocol", {}).get("training", {}).get("batch_size")) == 256]
    expected_training = {(seed, model) for seed in production_seeds for model in ("M1", "M2", "M3")}
    if (len(training) != 15 or len(completed) != 15
            or {(run.get("seed"), run.get("model")) for run in completed} != expected_training):
        missing.append(f"Completed full CUDA training: {len(completed)}/15 valid unique model/seed runs.")
    if report.get("execution_completed") is not True:
        missing.append("Full experiment execution has not been confirmed complete.")
    frequencies = report.get("frequency_runs", [])
    for model in _comparison_models(report):
        for regime in REGIMES:
            group = _group(frequencies, model=model, split="test_id_sub", regime=regime, method="S3b_diff")
            present = {record.get("seed") for record in group}
            if present != expected or len(group) != len(expected):
                missing.append(f"ID S3b_diff {model}/{regime}: {len(group)}/{len(expected)} seed records")
    scales = report.get("scale_runs", [])
    for model in ("M1", "M2", "M3"):
        for regime in ("T1", "T3"):
            group = _group(scales, model=model, split="test_id_sub", regime=regime)
            if {record.get("seed") for record in group} != expected or len(group) != len(expected):
                missing.append(f"ID {regime} scale {model}: {len(group)}/{len(expected)} seed records")
    portraits = report.get("portraits", [])
    for model in _comparison_models(report):
        if not _group(portraits, model=model, split="test_id_sub", seed=42):
            missing.append(f"ID T3 portrait {model}, seed 42")
    if "M0_Koopman" in _models(report):
        for key, expected_count in (("frequency_runs", 2000), ("scale_runs", 200),
                                    ("oos_scale_runs", 200), ("probe_training_runs", 50),
                                    ("predictor_diagnostics", 100), ("control_weight_checks", 10)):
            if len(report.get(key, [])) != expected_count:
                missing.append(f"Task 3.2 completeness {key}: {len(report.get(key, []))}/{expected_count}")
        grid = {(s, m, split, regime, method) for s in expected for m in _models(report)
                for split in _splits(report) for regime in REGIMES for method in METHODS}
        actual = [(r.get("seed"), r.get("model"), r.get("split"), r.get("regime"), r.get("method"))
                  for r in frequencies]
        if len(actual) != len(grid) or set(actual) != grid:
            missing.append("Task 3.2 frequency grid is incomplete or contains duplicate identities.")
        for key in ("scale_runs", "oos_scale_runs"):
            grid = {(s,m,split,regime) for s in expected for m in _models(report)
                    for split in _splits(report) for regime in ("T1","T3")}
            actual = [(r.get("seed"),r.get("model"),r.get("split"),r.get("regime"))
                      for r in report.get(key, [])]
            if len(actual) != len(grid) or set(actual) != grid:
                missing.append(f"Task 3.2 {key} grid is incomplete or contains duplicate identities.")
    return missing


def _absolute_table(report):
    rows = []
    for model in _models(report):
        for regime in REGIMES:
            group = _group(report.get("frequency_runs", []), split="test_id_sub",
                           model=model, regime=regime, method="S3b_diff")
            rows.append([model, regime, *[_moment(group, key) for key in
                         ("intercept", "q_median", "q_min", "q_max", "max_abs_q_error")]])
    return tabulate(rows, headers=["model", "regime", "intercept", "median q", "min q", "max q",
                                   "max absolute q error"], tablefmt="github")


def _oos_table(report):
    rows = []
    for split in _splits(report):
        for model in _models(report):
            for regime in ("T1", "T3"):
                group = _group(report.get("oos_scale_runs", []), split=split, model=model, regime=regime)
                rows.append([split, model, regime, f"{len(group)}/{len(_seeds(report))}",
                             _moment(group, "oos_phase_norm_r2"), _moment(group, "oos_axis_probe_r2"),
                             sum(r.get("status") != "completed" for r in group)])
    return tabulate(rows, headers=["test split", "model", "regime", "seeds", "OOS phase R²",
                                   "OOS axis R²", "invalid seeds"], tablefmt="github")


def _frequency_table(report, split):
    records = report.get("frequency_runs", [])
    rows = []
    for model in _models(report):
        for regime in REGIMES:
            for method in METHODS:
                group = _group(records, model=model, split=split, regime=regime, method=method)
                invalid = sum(int(record.get("invalid", 0) or 0) for record in group)
                rows.append([model, regime, method, f"{len(group)}/{len(_seeds(report))}",
                             _moment(group, "slope"), _moment(group, "slope_abs_error"),
                             _moment(group, "r2"), _moment(group, "one_minus_r2"), invalid])
    return tabulate(rows, headers=["model", "regime", "method", "seeds", "slope mean ± SD",
                                   "|slope+1.5|", "R²", "SSE/SST", "invalid orbits"], tablefmt="github")


def _scale_table(report):
    records = report.get("scale_runs", [])
    rows = []
    for split in _splits(report):
        for model in _models(report):
            for regime in ("T1", "T3"):
                group = _group(records, split=split, model=model, regime=regime)
                failed = sum(1 for record in group if record.get("error") or record.get("status") == "failed")
                constant = sum(record.get("state_norm_numerically_constant") is True for record in group)
                rows.append([split, model, regime, f"{len(group)}/{len(_seeds(report))}",
                             _moment(group, "phase_norm_r2"), _moment(group, "axis_probe_r2"),
                             _moment(group, "state_norm_r2"), constant, failed])
    return tabulate(rows, headers=["split", "model", "regime", "seeds", "phase-norm R²",
                                   "axis-probe R²", "state-norm R²", "constant-norm seeds", "failed seeds"],
                    tablefmt="github")


def _legacy_scale_table(report):
    rows = []
    for split in _splits(report):
        for model in _models(report):
            for regime in ("T1", "T3"):
                group = _group(report.get("scale_runs", []), split=split, model=model, regime=regime)
                rows.append([split, model, regime, _moment(group, "scale_r2_cycle_radius"),
                             _moment(group, "scale_r2_state_norm"), _moment(group, "energy_r2"),
                             _moment(group, "physical_state_norm_axis_r2"),
                             _moment(group, "cycle_radius_spearman")])
    return tabulate(rows, headers=["split", "model", "regime", "legacy cycle-radius R²",
                                   "legacy state-norm R²", "legacy energy R²", "physical norm→log(a) R²",
                                   "Spearman(s, log a)"], tablefmt="github")


def _factor_table(report, split):
    records = report.get("factor_frequency_runs", [])
    rows = []
    for regime in REGIMES:
        for method in METHODS:
            group = _group(records, model="M2", split=split, regime=regime, method=method)
            rows.append([regime, method, f"{len(group)}/{len(_seeds(report))}",
                         _moment(group, "slope_abs_error"), _moment(group, "r2"),
                         sum(int(record.get("invalid", 0) or 0) for record in group)])
    return tabulate(rows, headers=["regime", "method", "seeds", "leading factor |slope+1.5|",
                                   "leading factor R²", "invalid orbits"], tablefmt="github")


def _predictor_table(report):
    records = report.get("predictor_diagnostics", [])
    if not records:
        return "Pending: no predictor-invariant diagnostics are available."
    keys = ("orthogonality_max_abs", "norm_relative_drift_max", "action_log_drift_max",
            "ar_direct_relative_error_max", "temporal_displacement_ratio_median")
    rows = []
    for split in _splits(report):
        for model in _models(report):
            group = _group(records, split=split, model=model)
            if group:
                rows.append([split, model, f"{len(group)}/{len(_seeds(report))}",
                             *[_moment(group, key) for key in keys]])
    return tabulate(rows, headers=["split", "model", "seeds", "max |QᵀQ−I|",
                                   "max norm drift", "max log-action drift", "max AR/direct discrepancy",
                                   "median latent/input displacement ratio"], tablefmt="github")


def _plot_frequency_panel(axis, report):
    secondary = axis.twinx()
    x = np.arange(len(REGIMES))
    frequencies = report.get("frequency_runs", [])
    finite_errors, invalid_labels, zero_labels = [], [], []
    plot_data = {}
    for model in _comparison_models(report):
        errors, error_stds, logs, log_stds = [], [], [], []
        for regime in REGIMES:
            records = _group(frequencies, model=model, split="test_id_sub", regime=regime, method="S3b_diff")
            error = _finite_values(records, "slope_abs_error")
            ratio = _finite_values(records, "one_minus_r2")
            if len(error) and np.all(np.isfinite(error)) and np.all(error >= 0):
                errors.append(float(error.mean()))
                error_stds.append(float(error.std(ddof=1)) if len(error) > 1 else 0.0)
                finite_errors.extend(error[error > 0].tolist())
                if not np.all(error > 0):
                    zero_labels.append(f"{model}/{regime}: exact-zero slope error")
            else:
                errors.append(float("nan"))
                error_stds.append(float("nan"))
                invalid_labels.append(f"{model}/{regime}: undefined slope")
            if len(ratio) and np.all(np.isfinite(ratio)) and np.all(ratio > 0):
                log_values = np.log10(ratio)
                logs.append(float(log_values.mean()))
                log_stds.append(float(log_values.std(ddof=1)) if len(ratio) > 1 else 0.0)
            else:
                logs.append(float("nan"))
                log_stds.append(float("nan"))
                if len(ratio) and np.all(np.isfinite(ratio)) and np.all(ratio >= 0) and np.any(ratio == 0):
                    zero_labels.append(f"{model}/{regime}: zero SSE/SST (log = −∞)")
                else:
                    invalid_labels.append(f"{model}/{regime}: undefined residual")
        plot_data[model] = (np.asarray(errors), np.asarray(error_stds), np.asarray(logs), np.asarray(log_stds))
    zero_location = min(finite_errors) * 0.1 if finite_errors else 1e-16
    for offset, model in enumerate(_comparison_models(report)):
        errors, error_stds, logs, log_stds = plot_data[model]
        position = x + (offset - 1) * 0.045
        display_errors = np.where(errors == 0, zero_location, errors)
        axis.plot(position, display_errors, "-o", color=COLORS[model], linewidth=1.5, markersize=4)
        positive_band = np.isfinite(errors) & (errors - error_stds > 0)
        if np.any(positive_band):
            axis.fill_between(position[positive_band], errors[positive_band] - error_stds[positive_band],
                              errors[positive_band] + error_stds[positive_band], color=COLORS[model], alpha=0.09)
        secondary.plot(position, logs, "--s", color=COLORS[model], linewidth=1.25, markersize=3.5, alpha=0.85)
        finite = np.isfinite(logs) & np.isfinite(log_stds)
        if np.any(finite):
            secondary.fill_between(position[finite], logs[finite] - log_stds[finite],
                                   logs[finite] + log_stds[finite], color=COLORS[model], alpha=0.06)
    axis.set_yscale("log")
    axis.set_xticks(x, ("T1", "T2", "T3", "T4\n1–64", "T4\n1–512"))
    axis.set(xlabel="Evaluation regime", ylabel=r"$|\hat\beta_1+1.5|$ (solid)",
             title="(a) ID: phase differences, five seeds")
    secondary.set_ylabel(r"$\log_{10}(\mathrm{SSE}/\mathrm{SST})$ (dashed)")
    axis.grid(alpha=0.18)
    annotations = []
    if invalid_labels:
        annotations.append(f"Undefined aggregates: {len(invalid_labels)} metrics (see tables)")
    if zero_labels:
        annotations.append(f"Exact-zero metrics: {len(zero_labels)} (details in report/JSON)")
    if annotations:
        axis.text(0.02, -0.27, "\n".join(annotations), transform=axis.transAxes,
                  fontsize=7, va="top")
    handles = [Line2D([0], [0], color=COLORS[model], linewidth=2, label=model) for model in _comparison_models(report)]
    axis.legend(handles=handles, loc="best", fontsize=8, frameon=False)


def _plot_portraits(subgrid, figure, report):
    records = report.get("portraits", [])
    chosen = {}
    for model in _comparison_models(report):
        group = _group(records, model=model, split="test_id_sub", seed=42)
        if group:
            chosen[model] = group[0]
    identities = {(record.get("orbit_index"), record.get("a")) for record in chosen.values()}
    matched = len(identities) == 1 and len(chosen) == len(_comparison_models(report))
    for index, model in enumerate(_comparison_models(report)):
        axis = figure.add_subplot(subgrid[0, index])
        record = chosen.get(model, {})
        try:
            mode = np.asarray(record.get("U", []), dtype=np.float64)
            valid = mode.ndim == 2 and mode.shape[0] >= 2 and mode.shape[1] == 2 and np.all(np.isfinite(mode))
        except (ValueError, TypeError):
            mode, valid = None, False
        if valid:
            axis.plot(mode[:, 0], mode[:, 1], color=COLORS[model], linewidth=1.1)
            axis.scatter(mode[0, 0], mode[0, 1], s=17, color=COLORS[model], marker="o", zorder=3)
            axis.scatter(mode[-1, 0], mode[-1, 1], s=20, color=COLORS[model], marker="x", zorder=3)
            axis.set_box_aspect(1)
            axis.set_aspect("equal", adjustable="datalim")
        else:
            axis.text(0.5, 0.5, "Undefined / nonfinite\ntrajectory retained\nin detailed JSON", ha="center", va="center",
                      transform=axis.transAxes, fontsize=8)
        radius = _number(record.get("a"))
        label = f"a={radius:.3g}" if np.isfinite(radius) else "a unavailable"
        axis.set_title(model, fontsize=9, pad=18)
        if not matched:
            axis.text(0.04, 0.94, label, transform=axis.transAxes, ha="left", va="top", fontsize=7)
        axis.set_xlabel("PC1", fontsize=8)
        if index == 0:
            axis.set_ylabel("PC2 (own units)", fontsize=8)
        axis.tick_params(labelsize=6)
        axis.ticklabel_format(style="sci", axis="both", scilimits=(-2, 3), useMathText=True)
        axis.grid(alpha=0.15)
    figure.text(0.555, 0.87, "(b) Closed loop T3, seed 42", ha="center", fontsize=11)
    radius = _number(chosen.get("M0", {}).get("a"))
    match_text = f"Matched orbit a={radius:.3g}; " if matched and np.isfinite(radius) else "Orbit matching unavailable; "
    figure.text(0.555, 0.12, match_text + "own PCA axes and limits; ○ first, × last", ha="center", fontsize=7.5)


def _plot_scale_panel(axis, report):
    records = report.get("scale_runs", [])
    notes = []
    for model in ("M1", "M2", "M3"):
        group = _group(records, model=model, split="test_id_sub", regime="T1")
        axes, series, invalid = [], [], 0
        for record in group:
            try:
                x = np.asarray(record.get("physical_phase_norm", []), dtype=np.float64)
                s = np.asarray(record.get("r_rms", []), dtype=np.float64)
                valid = x.ndim == 1 and len(x) >= 2 and x.shape == s.shape and np.all(np.isfinite(x)) and np.all(np.isfinite(s))
            except (TypeError, ValueError):
                valid = False
            if valid:
                axes.append(x)
                series.append(s)
            else:
                invalid += 1
        if invalid or len(series) != len(_seeds(report)):
            notes.append(f"{model}: aggregate undefined ({invalid}/{len(group)} failed seeds)")
            continue
        if not all(x.shape == axes[0].shape and np.allclose(x, axes[0], atol=0, rtol=1e-12) for x in axes):
            notes.append(f"{model}: grids differ; raw seed curves shown")
            for x, s in zip(axes, series):
                order = np.argsort(x)
                axis.plot(x[order], s[order], color=COLORS[model], alpha=0.4, linewidth=0.8)
            continue
        values = np.stack(series)
        order = np.argsort(axes[0])
        x, mean = axes[0][order], values.mean(axis=0)[order]
        std = values.std(axis=0, ddof=1)[order] if len(values) > 1 else np.zeros_like(mean)
        axis.plot(x, mean, color=COLORS[model], label=model if model != "M3" else "M3 sphere", linewidth=1.6)
        axis.fill_between(x, mean - std, mean + std, color=COLORS[model], alpha=0.13)
    axis.set(xlabel=r"$\frac{1}{2}\log(a^2+\mu/a)$ (true phase log-norm)",
             ylabel=r"$r=\log(\mathrm{latent\ coordinate\ RMS})$",
             title="(c) H4 state-scale diagnostic, ID T1")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)
    if notes:
        axis.text(0.03, 0.03, "\n".join(notes), transform=axis.transAxes, fontsize=6.5, va="bottom")


def _plot(report, target):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 10, "savefig.dpi": 300, "axes.spines.top": False})
    figure = plt.figure(figsize=(22 if "M0_Koopman" in _models(report) else 18, 6.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.1, 1.35, 1.0], left=0.055, right=0.985,
                              bottom=0.22, top=0.8, wspace=0.45)
    _plot_frequency_panel(figure.add_subplot(grid[0, 0]), report)
    _plot_portraits(grid[0, 1].subgridspec(1, len(_comparison_models(report)), wspace=0.75), figure, report)
    _plot_scale_panel(figure.add_subplot(grid[0, 2]), report)
    figure.suptitle("Task 3.2: OPF remediation and frozen Koopman control", fontsize=14, y=0.97)
    figure.text(0.5, 0.92, "Full 64D estimator S3b_diff; T1/T2/T3: 512 points; T4: trained 1–64 and extended 1–512; Δt = 0.04",
                ha="center", fontsize=9)
    figure.text(0.5, 0.055, "Bands: ±1 seed SD. Undefined seeds remain undefined in aggregates; all methods and ID/OOD outcomes are reported in JSON.",
                ha="center", fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300, facecolor="white", metadata={"Software": "kepler-jepa Exp2"})
    plt.close(figure)


def _markdown(report, missing, figure_written):
    checks = report.get("checks", [])
    training = report.get("training_runs", [])
    failed = [check for check in checks if check.get("passed") is False]
    pending = [check for check in checks if check.get("passed") is None]
    passed = [check for check in checks if check.get("passed") is True]
    completed_training = [run for run in training if run.get("status") == "completed"]
    execution = report.get("status", "unknown")
    smoke = report.get("protocol", {}).get("smoke_test", False)
    acceptance_text = ("not evaluated (engineering smoke only)" if smoke else
                       f"{len(passed)}/{len(checks)} passed, {len(failed)} failed, {len(pending)} pending")
    sections = ["# Task 3.2: OPF remediation and scientific controls", "",
                f"Execution status: **{execution}**. Completed training records: **{len(completed_training)}/{len(training)}**. "
                f"Scientific criteria: **{acceptance_text}**.", "",
                "Execution completion and scientific acceptance are separate outcomes. A completed run can fail "
                "a prespecified hypothesis; numerical results are retained without tuning them to pass.", ""]
    if smoke:
        sections.extend(["**CPU ENGINEERING SMOKE ONLY. These shortened runs do not test the scientific hypotheses "
                         "or satisfy the requested GPU experiment.**", ""])
    if missing:
        sections.extend(["## Pending publication figure", "",
                         "GPU experiments or required evaluation records are incomplete. No final publication figure "
                         "was generated from missing results.", "",
                         *[f"- {item}" for item in missing], ""])
    elif figure_written:
        sections.extend(["![Experiment 2 circular regimes](figures/exp2_circular_regimes.png)", ""])
    sections.extend(["## Protocol and numerical settings", "", "```json",
                     json.dumps(_plain_json(report.get("protocol", {})), indent=2, ensure_ascii=False), "```", "",
                     "T1 encodes true future windows. T2 predicts one step from each true predecessor. T3 repeatedly "
                     "uses horizon 1 from one initial window. T4 predicts trained horizons 1–64 from that "
                     "same initial window; T4_512 separately evaluates horizons 1–512. All sequences exclude z0 and align "
                     "to future window starts. Each orbit/regime uses independent centered, unwhitened PCA for the "
                     "primary frequency estimates. Frequency estimation always uses the full 64D latent vector; "
                     "the M2 leading-factor result below is a separate diagnostic.", "",
                     "In the full protocol, δ=4 and simulator dt=0.01 give 512 points separated by 0.04. "
                     "T1/T2/T3 and T4_512 span **20.44** between first and last samples and reach **20.48** from the "
                     "initial window. Trained T4 spans **2.52** (last horizon **2.56**). The 2100 raw samples supply "
                     "all required true windows without padding. ID orbits cover more than one period; long-period "
                     "OOD orbits can still be partial arcs. FFT resolution and nonlinear projection geometry remain relevant.", "",
                     "An odd encoder maps opposite physical states to opposite latents. This gives zero full-cycle "
                     "mean for antipodally symmetric circular trajectories, but does **not** give zero sample mean on "
                     "an arbitrary finite arc. Differencing removes a constant origin shift; it does not remove "
                     "anisotropic or nonlinear phase distortion.", "",
                     "Training uses float32; the same learned numeric parameters are promoted to float64 for all "
                     "all model evaluations. The direct prediction loss is normalized per orbit by target latent energy, "
                     "and Task 3.2 also normalizes AR state and increment losses per orbit for M1/M2. M3 retains its Task 3.1 objective. The intra-orbit "
                     "temporal displacement/scale term has weight 1.0 for M1/M2. The JSON records exact settings.", "",
                     "## Training execution", ""])
    if training:
        rows = [[run.get("seed", "?"), run.get("model", "?"), run.get("status", "recorded"),
                 run.get("completed_steps", run.get("steps_completed", run.get("steps", "see JSON"))), run.get("device", "see protocol"),
                 run.get("elapsed_seconds", "see JSON")] for run in training]
        sections.extend([tabulate(rows, headers=["seed", "model", "status", "steps", "device", "seconds"], tablefmt="github"), ""])
    else:
        sections.extend(["Pending: no completed GPU training records are available.", ""])
    sections.extend(["## Frequency regressions", "",
                     "Entries are mean ± sample standard deviation over the recorded seeds. The seed count is explicit. "
                     "An undefined seed makes the aggregate undefined; no `nanmean`, invalid-orbit filtering, or "
                     "best-seed selection is used. SSE/SST is computed directly rather than by subtracting rounded R².", ""])
    for split in _splits(report):
        sections.extend([f"### {split}", "", _frequency_table(report, split), ""])
    sections.extend(["## Absolute ID frequency (S3b_diff)", "", _absolute_table(report), "",
                     "q=omega_hat/a^(-1.5), mu=1. Each table cell is a mean ± seed SD; q_min/q_max "
                     "are per-seed orbit extrema before aggregation. Raw per-orbit q remains in JSON. "
                     "One invalid orbit leaves that seed's aggregate q undefined.", "",
                     "## Out-of-sample H4 (train to test)", "", _oos_table(report), "",
                     "Coefficients and constant-column selection are fitted on all 200 training radii, separately "
                     "for T1 and T3, and applied unchanged to each test split. Negative test R² is retained. "
                     "OOS phase R² uses the disclosed zero-score convention for a numerically constant response. "
                     "The probe labels use physical metadata; these labels never enter JEPA training or the predictor.", ""])
    sections.extend(["## Dependence of T3 and T4_512", "",
                     "For the action-angle predictor, rates depend only on plane radii preserved by each rotation. "
                     "Consequently P_1 composed k times equals P_k in exact arithmetic. The two sequences are "
                     "computed separately, but their frequency regressions are not independent evidence. Their "
                     "agreement tests numerical implementation; arbitrary wrong rates have the same property. "
                     "The frozen M0_Koopman control separates this structural property from learned physical accuracy.", ""])
    sections.extend(["## M2 leading-factor diagnostic", "",
                     "The leading factor is selected per orbit and regime by temporal variance without radius/frequency "
                     "labels. This is supplementary to the full 64D primary analysis and never replaces failed full-vector "
                     "criteria. Each record retains `active_factor_indices` for all orbits.", ""])
    for split in _splits(report):
        sections.extend([f"### {split}", "", _factor_table(report, split), ""])
    sections.extend(["## H4: phase-scale and axis probes", "", _scale_table(report), "",
                     "`phase_norm_r2` regresses log latent coordinate RMS **as the response** on "
                     "0.5 log(a²+μ/a). A constant factor in the true physical RMS changes only the intercept. "
                     "`axis_probe_r2` reconstructs log(a) from the pair [log state coordinate RMS, log RMS(diff(Z)/0.04)]. "
                     "`state_norm_r2` is the legacy log(a)-on-state-norm score, included under both names.", "",
                     "A predictor with log-feature standard deviation ≤1e−6 is treated as numerically constant by "
                     "default; the threshold, feature variation and dropped columns are retained in each JSON fit. "
                     "A constant predictor against varying axes gives R²=0. Conventional R² is undefined for an exactly "
                     "constant response; Task 3.1 explicitly assigns `phase_norm_r2=0` when latent norm response std≤1e−6. "
                     "The JSON flags this zero-score convention and retains the conventional fit value.", "",
                     "### Preserved Task 3 metrics and physical control", "", _legacy_scale_table(report), "",
                     "The physical norm→log(a) R² is computed on each actual radius grid, rather than hardcoding a "
                     "theoretical ceiling. It measures the nonmonotonic phase-norm confound in the previous one-feature "
                     "probe. Legacy s/r average coordinate-summed squared norms over time; new r_rms/v_rms average "
                     "over time and coordinates. Their dimensional constant offsets do not change linear-probe R².", "",
                     "Spearman(s, log a) supplies the direction and rank diagnostic: high linear R² alone does not "
                     "establish increasing monotonicity. The energy target log|E|=log(μ/2)−log(a) is affine in log(a), "
                     "so energy R² is a related scale probe rather than independent evidence of learned conservation.", "",
                     "Unit state norm does not imply a constant centered cycle radius on a finite arc. A normalized "
                     "state can retain radius information in its direction and its temporal velocity. Zero M3 state/phase "
                     "norm scores therefore establish only loss of the norm channel; they do not establish total loss of "
                     "scale information, and a successful M3 axis probe would be reported without contradiction. "
                     "Low R² of these linear norm probes also does not establish absence of nonlinear scale information "
                     "elsewhere in the representation. The M3 normalization result is therefore measured, not assumed.", "",
                     "## Predictor invariants and temporal displacement", "", _predictor_table(report), "",
                     "Drift and AR/direct discrepancy are measured on the actual float64 generated trajectories. "
                     "The orthogonal architecture supplies a mathematical conservation mechanism; finite precision "
                     "deviations and learned-frequency accuracy are separate empirical quantities. Temporal displacement "
                     "ratios compare latent and input coordinate RMS at the prescribed within-orbit lag.", "",
                     "## Prespecified scientific acceptance", "",
                     "There are 16 inherited criteria plus two M0_Koopman controls in Task 3.2. Each T4 model criterion jointly checks five seeds at trained horizons "
                     "1–64 and the same five seeds at extended horizons 1–512: **10 seed/horizon records**. T3 and T1 "
                     "criteria have five seed records. The extra T4 window does not replace or average away the trained "
                     "window. All thresholds are checked per seed.", ""])
    if checks:
        rows = [[check.get("criterion", "?"), "NOT EVALUATED (smoke)" if smoke else "PASS" if check.get("passed") is True else "FAIL" if check.get("passed") is False else "PENDING",
                 (f"{check.get('n_seed_records')}/{check.get('expected_seed_records')}"
                  if "n_seed_records" in check else "see details"),
                 json.dumps(_plain_json(check.get("details", {})), ensure_ascii=False)] for check in checks]
        sections.extend([tabulate(rows, headers=["criterion", "status", "seed/horizon records", "details"], tablefmt="github"), ""])
    else:
        sections.extend(["Pending: scientific criteria have not yet been evaluated.", ""])
    scale_failures = [record for record in report.get("scale_runs", []) if record.get("error")]
    if scale_failures:
        sections.extend(["### Failed scale diagnostics", "", tabulate(
            [[record.get("seed"), record.get("model"), record.get("split"), record.get("regime"), record.get("error")]
             for record in scale_failures], headers=["seed", "model", "split", "regime", "error"], tablefmt="github"), ""])
    sections.extend(["## Environment", "", "```json",
                     json.dumps(_plain_json(report.get("environment", {})), indent=2, ensure_ascii=False), "```", "",
                     "## Artifacts", "", "- [Full stdout](exp2_circular_stdout.txt)",
                     "- [All per-orbit and per-seed results](exp2_circular_results.json)",
                     "- [300 dpi publication figure](figures/exp2_circular_regimes.png)" if figure_written else "- Publication figure pending complete GPU evaluation records.", ""])
    return "\n".join(sections)


def write_report(report, output_dir):
    """Write Markdown and, for complete actual records, the 300 dpi figure.

    Return paths as {markdown: Path, figure: Path | None}. This helper never
    writes the orchestration JSON or stdout and never invents missing outcomes.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    missing = _figure_missing(report)
    figure = None
    if not missing:
        figure = output / "figures" / "exp2_circular_regimes.png"
        _plot(report, figure)
    markdown = output / "exp2_circular_report.md"
    markdown.write_text(_markdown(report, missing, figure is not None), encoding="utf-8")
    return {"markdown": markdown, "figure": figure}
