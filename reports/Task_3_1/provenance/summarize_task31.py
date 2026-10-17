"""Reproduce descriptive Task 3.1 statistics from its immutable GPU JSON.

Run from the repository root:
  python reports/Task_3_1/provenance/summarize_task31.py > task31_summary.json
An optional positional argument selects another results JSON. No training,
selection of successful seeds, or modification of acceptance criteria occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = list(range(42, 47))
REGIMES = ("T1", "T3", "T4", "T4_512")


def stats(values):
    a = np.array([np.nan if v is None else v for v in values], dtype=float)
    finite = bool(np.isfinite(a).all())
    return {"n": len(a), "undefined": int((~np.isfinite(a)).sum()),
            "mean": float(a.mean()) if finite else None,
            "sample_sd": float(a.std(ddof=1)) if finite and len(a) > 1 else None,
            "min": float(a.min()) if finite else None,
            "max": float(a.max()) if finite else None}


def select(rows, model, split, regime, method=None):
    chosen = sorted([r for r in rows if r["model"] == model
                     and r["split"] == split and r["regime"] == regime
                     and (method is None or r["method"] == method)],
                    key=lambda r: r["seed"])
    if [r["seed"] for r in chosen] != SEEDS:
        raise ValueError(f"Incomplete/duplicate seed grid: {model}/{split}/{regime}")
    return chosen


def ratios(row):
    a = np.array(row["semi_major_axes"], dtype=float)
    w = np.array([np.nan if v is None else v for v in row["omega_est"]])
    return w / a ** -1.5


def fit_group(row, lower):
    a = np.array(row["semi_major_axes"], dtype=float)
    w = np.array([np.nan if v is None else v for v in row["omega_est"]])
    mask = a < .5 if lower else a > 5
    a, w = a[mask], w[mask]
    if len(a) != 25:
        raise ValueError("Expected all 25 pre-existing OOD radii in each group")
    invalid = ~np.isfinite(w) | (w <= 0)
    q = w / a ** -1.5
    if invalid.any():
        slope, intercept, r2 = None, None, None
    else:
        x, y = np.log(a), np.log(w)
        xc, yc = x - x.mean(), y - y.mean()
        slope = float(xc @ yc / (xc @ xc))
        intercept = float(y.mean() - slope * x.mean())
        sst = float(yc @ yc)
        r2 = float(1 - np.square(y - (intercept + slope*x)).sum()/sst) if sst else None
    return {"seed": row["seed"], "n": len(a), "invalid": int(invalid.sum()),
            "slope": slope, "intercept": intercept, "r2": r2,
            "ratio_min": float(q.min()) if not invalid.any() else None,
            "ratio_max": float(q.max()) if not invalid.any() else None}


def summarize(path):
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    out = {"results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "elapsed_seconds": report["elapsed_seconds"],
           "execution_completed": report["execution_completed"],
           "acceptance_passed": report["acceptance_passed"],
           "aggregation": "all five seeds, sample SD ddof=1; undefined retained",
           "frequency": {}, "scale": {}, "factor_frequency": {},
           "factor_selection_counts": {}, "m1_absolute_id_frequency": [],
           "ood_t3_subgroups": {}, "predictor_diagnostics": {},
           "last_training_batch": {}, "estimator_invalid": {}}
    for split in ("test_id_sub", "test_ood"):
        for model in ("M0", "M1", "M2", "M3"):
            for regime in REGIMES:
                rows = select(report["frequency_runs"], model, split, regime, "S3b_diff")
                key = f"{split}/{model}/{regime}"
                out["frequency"][key] = {
                    k: stats([r[k] for r in rows])
                    for k in ("slope", "slope_abs_error", "r2", "intercept")}
                out["frequency"][key]["invalid_orbits"] = sum(r["invalid"] for r in rows)
                out["frequency"][key]["n_orbits"] = sum(r["n_orbits"] for r in rows)
                out["frequency"][key]["seeds_passing_recovery_threshold"] = [
                    r["seed"] for r in rows if r["invalid"] == 0
                    and r["slope_abs_error"] is not None and r["r2"] is not None
                    and r["slope_abs_error"] < .05 and r["r2"] > .99]
            for regime in ("T1", "T3"):
                rows = select(report["scale_runs"], model, split, regime)
                out["scale"][f"{split}/{model}/{regime}"] = {
                    k: stats([r[k] for r in rows]) for k in (
                        "phase_norm_r2", "axis_probe_r2", "state_norm_r2",
                        "scale_r2_cycle_radius", "scale_r2_state_norm", "energy_r2",
                        "physical_state_norm_axis_r2")}
    for regime in REGIMES:
        rows = select(report["factor_frequency_runs"], "M2", "test_id_sub", regime, "S3b_diff")
        out["factor_frequency"][regime] = {
            k: stats([r[k] for r in rows]) for k in ("slope", "slope_abs_error", "r2")}
        out["factor_frequency"][regime]["invalid_orbits"] = sum(r["invalid"] for r in rows)
        counts = Counter(i for r in rows for i in r["active_factor_indices"])
        out["factor_selection_counts"][regime] = [counts[i] for i in range(4)]
    rows3 = select(report["frequency_runs"], "M1", "test_id_sub", "T3", "S3b_diff")
    rows4 = select(report["frequency_runs"], "M1", "test_id_sub", "T4_512", "S3b_diff")
    for r, r4 in zip(rows3, rows4):
        q = ratios(r)
        if not np.isfinite(q).all():
            raise ValueError("Invalid M1 ID frequency; do not filter before ratios")
        out["m1_absolute_id_frequency"].append({
            "seed": r["seed"], "intercept": r["intercept"],
            "ratio_median": float(np.median(q)), "ratio_min": float(q.min()),
            "ratio_max": float(q.max()), "max_relative_error": float(np.abs(q-1).max()),
            "max_t3_t4_512_ratio_difference": float(np.abs(q-ratios(r4)).max())})
    for model in ("M1", "M2", "M3"):
        rows = select(report["frequency_runs"], model, "test_ood", "T3", "S3b_diff")
        for lower in (True, False):
            fits = [fit_group(r, lower) for r in rows]
            out["ood_t3_subgroups"][f"{model}/{'inner' if lower else 'outer'}"] = {
                "per_seed": fits, "slope": stats([r["slope"] for r in fits]),
                "r2": stats([r["r2"] for r in fits]),
                "ratio_min": stats([r["ratio_min"] for r in fits])["min"],
                "ratio_max": stats([r["ratio_max"] for r in fits])["max"]}
    for model in ("M0", "M1", "M2", "M3"):
        rows = [r for r in report["predictor_diagnostics"] if r["model"] == model]
        if len(rows) != 10:
            raise ValueError("Expected two splits times five seeds for diagnostics")
        fields = ("norm_relative_drift_max", "action_log_drift_max",
                  "ar_direct_relative_error_max", "orthogonality_max_abs",
                  "temporal_displacement_ratio_median")
        out["predictor_diagnostics"][model] = {
            "both_splits": {k: stats([r[k] for r in rows]) for k in fields},
            "id": {k: stats([r[k] for r in rows if r["split"] == "test_id_sub"])
                   for k in fields}}
    for model in ("M1", "M2", "M3"):
        histories = [r["history"][-1] for r in report["training_runs"] if r["model"] == model]
        if len(histories) != 5 or any(h["step"] != 2000 for h in histories):
            raise ValueError("Expected all five training histories at step 2000")
        out["last_training_batch"][model] = {
            k: stats([r[k] for r in histories])
            for k in ("total", "direct", "ar", "temporal", "orthogonality", "isotropic")}
    for method in report["protocol"]["frequency_methods"]:
        rows = [r for r in report["frequency_runs"] if r["method"] == method]
        out["estimator_invalid"][method] = {
            "invalid_orbits": sum(r["invalid"] for r in rows),
            "undefined_regressions": sum(r["slope"] is None for r in rows)}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1]/"exp2_circular_results.json")
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), ensure_ascii=False, indent=2, allow_nan=False))
