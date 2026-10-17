"""Task 2 calibration of the four Task 1 scientific checks.

Run ``python -m tests.verify_task1``. All requested thresholds are applied
literally; scientifically unmet criteria return exit 1 rather than being
relaxed. The original Task 1 reports are preserved separately.
"""

from __future__ import annotations

import argparse
from functools import partial
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate

from src.dataset import (build_dataset_circ, build_dataset_counterfactual,
                         build_dataset_ellip)
from src.simulator import (DT, compute_orbital_invariants,
                           generate_kepler_analytical, integrate_verlet)
from src.spectral import (evaluate_kepler_regression, freq_s1_discrete_fft,
                          freq_s2_interpolated_fft, freq_s3a_dmd_koopman,
                          freq_s3b_phase_unwrap, project_to_2d_mode)

METHODS = {"S1": freq_s1_discrete_fft, "S2": freq_s2_interpolated_fft,
           "S3a_std": freq_s3a_dmd_koopman, "S3b_std": freq_s3b_phase_unwrap,
           "S3a_diff": partial(freq_s3a_dmd_koopman, use_differences=True),
           "S3b_diff": partial(freq_s3b_phase_unwrap, use_differences=True)}


def _table(headers: list, rows: list) -> None:
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".9g"))
    print()


def _estimate(trajectories: np.ndarray, method: str, centered: bool = False) -> np.ndarray:
    result = []
    for trajectory in trajectories:
        coordinates = trajectory[:, :2]
        if centered:
            coordinates = project_to_2d_mode(coordinates)
        try:
            result.append(METHODS[method](coordinates, DT))
        except ValueError:
            result.append(np.nan)
    return np.asarray(result, dtype=np.float64)


def _regression(a: np.ndarray, frequencies: np.ndarray) -> dict:
    # Never drop unsuccessful orbits to make a complete regression pass.
    invalid = int(np.count_nonzero(~np.isfinite(frequencies) | (frequencies <= 0)))
    if invalid:
        return {"slope": np.nan, "slope_abs_error": np.nan, "r2": np.nan,
                "one_minus_r2": np.nan, "log10_one_minus_r2": np.nan, "invalid": invalid}
    return {**evaluate_kepler_regression(a, frequencies), "invalid": 0}


def _identity_r2(truth: np.ndarray, estimate: np.ndarray, *, log: bool = False) -> float:
    """R2 against the identity, without refitting a slope or intercept."""
    if np.any(~np.isfinite(estimate)) or np.any(estimate <= 0):
        return float("nan")
    y, prediction = (np.log(truth), np.log(estimate)) if log else (truth, estimate)
    return float(1 - np.sum((prediction - y)**2) / np.sum((y - y.mean())**2))


def _check(checks: list, label: str, passed: bool) -> None:
    checks.append([label, "PASS" if passed else "FAIL"])


def test1(seeds: list[int], checks: list, results: dict) -> None:
    print("TEST 1: analytical vs Verlet substeps=1 and 5; N=50, T=1024, dt=.01")
    rows, records = [], []
    analytical_pass, numerical_pass, reversals_pass = True, {1: True, 5: True}, True
    for seed in seeds:
        for variant, batch in build_dataset_ellip("test_id", seed).items():
            states = batch["trajectories"]
            exact_inv = compute_orbital_invariants(states)
            exact_max = float(exact_inv["energy_drift"].max())
            analytical_pass &= exact_max < 1e-13
            full_cycle = batch["meta"]["omega_true"] * ((states.shape[1]-1)*DT) >= 2*np.pi
            for substeps, threshold in ((1, 3e-3), (5, 1.5e-4)):
                verlet = integrate_verlet(states[:, 0, :2], states[:, 0, 2:],
                                          states.shape[1], substeps=substeps)
                inv = compute_orbital_invariants(verlet)
                num_max = float(inv["energy_drift"].max())
                changes = np.diff(inv["energy"], axis=1)
                tol = 100 * np.finfo(np.float64).eps * np.abs(inv["energy"][:, :1])
                reversals = np.any(changes > tol, axis=1) & np.any(changes < -tol, axis=1)
                eligible = int(full_cycle.sum())
                reversed_count = int((reversals & full_cycle).sum())
                numerical_pass[substeps] &= num_max < threshold
                reversals_pass &= reversed_count == eligible
                rows.append([seed, variant, substeps, exact_max, num_max,
                             float(inv["angular_momentum_drift"].max()),
                             f"{reversed_count}/{eligible}"])
                records.append({"seed": seed, "variant": variant, "substeps": substeps,
                                "a": batch["meta"]["a"].tolist(),
                                "e": batch["meta"]["e"].tolist(),
                                "analytical_energy_drift": exact_inv["energy_drift"].tolist(),
                                "verlet_energy_drift": inv["energy_drift"].tolist(),
                                "full_cycle": full_cycle.tolist(),
                                "energy_sign_reversals": reversals.tolist()})
    _table(["seed", "start", "substeps", "exact max dE/E", "Verlet max dE/E",
            "Verlet max dL/L", "sign reversals /full orbits"], rows)
    print("Sign reversals establish observed non-monotonic energy error; they do not")
    print("prove absence of secular drift at arbitrarily long integration times.\n")
    _check(checks, "T1 analytical energy <1e-13", analytical_pass)
    _check(checks, "T1 Verlet substeps=1 energy <3e-3", numerical_pass[1])
    _check(checks, "T1 Verlet substeps=5 energy <1.5e-4", numerical_pass[5])
    _check(checks, "T1 energy reversals on 100% of full-period orbits", reversals_pass)
    results["test1"] = records


def test2(seeds: list[int], checks: list, results: dict) -> None:
    print("TEST 2: circles, test_id radii restricted to [.5,2], T_eval=256 and 4096")
    print("raw = physical force-center origin; centered = project_to_2d_mode control.")
    print("Standard S3 long-window accuracy is checked at its specified physical origin;")
    print("centered standard S3 is reported separately because temporal centering changes it.")
    rows, records = [], []
    s1_short, diff_short = True, True
    s1_long, s2_long, s3_long = True, True, True
    for seed in seeds:
        meta = build_dataset_circ("test_id", seed)["meta"]
        selected = meta["a"] <= 2.0
        a = meta["a"][selected]
        states = generate_kepler_analytical(a[:, None], np.zeros((len(a), 1)),
                    np.arange(4096, dtype=np.float64)[None, :] * DT,
                    M0=meta["M0"][selected, None])
        for length in (256, 4096):
            for origin in ("raw", "centered"):
                for method in METHODS:
                    frequencies = _estimate(states[:, :length], method, centered=origin == "centered")
                    reg = _regression(a, frequencies)
                    if length == 256 and origin == "raw" and method == "S1":
                        s1_short &= len(np.unique(frequencies)) == 1 and np.isnan(reg["r2"])
                    if length == 256 and origin == "centered" and method.endswith("_diff"):
                        diff_short &= reg["slope_abs_error"] < 1e-10 and reg["one_minus_r2"] < 1e-20
                    if length == 4096 and origin == "raw":
                        if method == "S1":
                            s1_long &= 1e-5 < reg["one_minus_r2"] < 5e-2
                        elif method == "S2":
                            s2_long &= reg["r2"] > .9999
                        elif method in ("S3a_std", "S3b_std"):
                            s3_long &= reg["one_minus_r2"] < 1e-7
                    rows.append([seed, length, origin, method, len(a), reg["slope"],
                                 reg["one_minus_r2"], reg["invalid"]])
                    records.append({"seed": seed, "samples": length, "origin": origin,
                                    "method": method, "a": a.tolist(), "omega": frequencies.tolist(),
                                    "coverage_periods": (a**-1.5 * ((length-1)*DT) / (2*np.pi)).tolist(),
                                    **reg})
    _table(["seed", "T", "origin", "method", "N", "slope", "1-R2 (SSE/SST)", "invalid"], rows)
    print("A constant short-window S1 bin has zero log-frequency variance: R2 is undefined.")
    print("Difference methods annihilate translations; no mean is removed after differencing.\n")
    _check(checks, "T2 short S1 all frequencies in one bin (R2 undefined)", s1_short)
    _check(checks, "T2 centered short diff-S3 slope error <1e-10, 1-R2 <1e-20", diff_short)
    _check(checks, "T2 long S1 1e-5 < 1-R2 <5e-2", s1_long)
    _check(checks, "T2 long S2 R2 >.9999", s2_long)
    _check(checks, "T2 long physical-origin S3a/S3b 1-R2 <1e-7", s3_long)
    results["test2"] = records


def test3(seeds: list[int], checks: list, results: dict) -> None:
    print("TEST 3: elliptic raw phase, L=64; random e in [.2,.6]")
    print("The dataset centers its initial L=16 context at an apsis; L=64 is the first 64 samples.")
    rows, records = [], []
    inequalities_pass, slope_pass = True, True
    for seed in seeds:
        for variant, batch in build_dataset_ellip("test_id", seed).items():
            a, n = batch["meta"]["a"], batch["meta"]["omega_true"]
            for length in (64, 1024):
                frequencies = _estimate(batch["trajectories"][:, :length], "S3b_std")
                reg = _regression(a, frequencies)
                ratio = frequencies / n
                if length == 64:
                    expected = frequencies > n if variant == "perihelion_init" else frequencies < n
                    inequalities_pass &= bool(np.all(expected))
                    slope_pass &= -1.75 <= reg["slope"] <= -1.10
                rows.append([seed, variant, length, reg["slope"], reg["r2"],
                             float(ratio.min()), float(ratio.max())])
                records.append({"seed": seed, "variant": variant, "samples": length,
                                "omega": frequencies.tolist(), "omega_mean": n.tolist(), **reg})
    _table(["seed", "start", "T", "slope", "R2", "min omega/n", "max omega/n"], rows)
    _check(checks, "T3 L=64 peri >n> ap for 50/50 orbits", inequalities_pass)
    _check(checks, "T3 L=64 slope in [-1.75,-1.10]", slope_pass)
    results["test3"] = records


def test4(seeds: list[int], checks: list, results: dict) -> None:
    print("TEST 4: counterfactual, all N=50 retained, T=16384, substeps=5, dt=.01")
    print("Long acceptance uses log-space identity R2; frequency-space and refitted R2 are diagnostic.")
    print("No orbit selection or true-period correction is used by the estimators.")
    rows, coverage_rows, records = [], [], []
    short_log_pass, short_freq_pass, long_s2_pass, long_phase_pass = True, True, True, True
    for seed in seeds:
        batch = build_dataset_counterfactual(seed)
        initial, meta = batch["trajectories"][:, 0], batch["meta"]
        states = integrate_verlet(initial[:, :2], initial[:, 2:], 16384, substeps=5)
        cycles = meta["omega_true"] * ((states.shape[1]-1)*DT) / (2*np.pi)
        coverage_rows.append([seed, float(cycles.min()), float(cycles.max()),
                              int(np.count_nonzero(cycles < 1)), int(np.count_nonzero(cycles < 2))])
        for length, method, reference in ((16, "S3b_std", "omega_init"),
                                         (16384, "S2", "omega_true"),
                                         (16384, "S3b_std", "omega_true")):
            frequencies = _estimate(states[:, :length], method)
            truth = meta[reference]
            identity = _identity_r2(truth, frequencies)
            log_identity = _identity_r2(truth, frequencies, log=True)
            fitted = _regression(truth, frequencies)["r2"]
            relative = np.abs(frequencies / truth - 1)
            rows.append([seed, length, method, reference, identity, log_identity, fitted,
                         float(np.max(relative))])
            if length == 16:
                short_log_pass &= log_identity > .99995
                short_freq_pass &= identity > .9998
            elif method == "S2":
                long_s2_pass &= log_identity > .999
            else:
                long_phase_pass &= log_identity > .999
            records.append({"seed": seed, "samples": length, "substeps": 5,
                            "method": method, "origin": "raw", "N": len(truth),
                            "reference": reference, "a_true": meta["a_true"].tolist(),
                            "coverage_periods": (meta["omega_true"]*((length-1)*DT)/(2*np.pi)).tolist(),
                            "omega_est": frequencies.tolist(), "omega_reference": truth.tolist(),
                            "R2_identity_frequency": identity, "R2_identity_log": log_identity,
                            "R2_refitted_log": fitted, "relative_errors": relative.tolist()})
    _table(["seed", "T", "method", "reference", "R2_frequency", "R2_log", "R2_refit_log", "max rel err"], rows)
    _table(["seed", "min periods", "max periods", "<1 period /50", "<2 periods /50"], coverage_rows)
    _check(checks, "T4 short phase R2_identity_log >.99995", short_log_pass)
    _check(checks, "T4 short phase R2_identity_frequency >.9998", short_freq_pass)
    _check(checks, "T4 all-orbit long S2 R2_identity_log >.999", long_s2_pass)
    _check(checks, "T4 all-orbit long S3b R2_identity_log >.999", long_phase_pass)
    results["test4"] = records


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _aggregates(results: dict) -> None:
    rows = []
    for length in (256, 4096):
        for origin in ("raw", "centered"):
            for method in METHODS:
                group = [r for r in results["test2"] if r["samples"] == length
                         and r["origin"] == origin and r["method"] == method]
                slopes = np.array([r["slope"] for r in group])
                ratios = np.array([r["one_minus_r2"] for r in group])
                rows.append([length, origin, method, slopes.mean(),
                             slopes.std(ddof=1) if len(slopes)>1 else 0., ratios.mean(),
                             ratios.std(ddof=1) if len(ratios)>1 else 0.])
    print("TEST 2 ACROSS-SEED SUMMARY (sample std, ddof=1; undefined values remain NaN)")
    _table(["T", "origin", "method", "slope mean", "slope std", "1-R2 mean", "1-R2 std"], rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--json", type=Path, help="Optional detailed machine-readable results")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct nonnegative integers")
    environment = {name: importlib.metadata.version(name) for name in
                   ("numpy", "scipy", "scikit-learn", "torch", "matplotlib", "tabulate")}
    print("Kepler-JEPA Task 2 / Part A calibrated scientific diagnostics")
    print("Command: python -m tests.verify_task1" + (" " + " ".join(sys.argv[1:]) if sys.argv[1:] else ""))
    print(f"Python {platform.python_version()}; {platform.system()}; seeds={args.seeds}")
    print("Dependencies: " + ", ".join(f"{name}={version}" for name, version in environment.items()))
    print("dtype=float64; mu=1; dt=.01; substeps are explicit; no frequency calibration.\n")
    checks, results = [], {"task": "task2_calibration", "environment": environment,
                          "seeds": args.seeds, "dtype": "float64", "dt": DT}
    for test in (test1, test2, test3, test4):
        test(args.seeds, checks, results)
    print("ACCEPTANCE SUMMARY (all checks run before deciding exit status)")
    _table(["criterion", "status"], checks)
    _aggregates(results)
    passed = all(row[1] == "PASS" for row in checks)
    results["checks"], results["acceptance_passed"] = checks, passed
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Overall acceptance: " + ("PASS (exit 0)" if passed else "NOT MET (exit 1)"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
