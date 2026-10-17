"""Independent Task 3.1 transfer/completeness audit, not a scientific PASS test.

Usage: python audit_gpu_artifacts_task31.py DOWNLOAD_ROOT REPOSITORY OUTPUT_JSON
DOWNLOAD_ROOT contains reports/Task_3_1 as extracted from the IDH bundle. Source
files and the preserved Task 3 archive are checked against REPOSITORY. Nothing
is written there unless the caller explicitly chooses OUTPUT_JSON inside it.
Checkpoint loading is restricted to weights_only=True and map_location='cpu'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

from PIL import Image
import torch


SEEDS = tuple(range(42, 47))
MODELS = ("M0", "M1", "M2", "M3")
SPLITS = ("test_id_sub", "test_ood")
REGIMES = ("T1", "T2", "T3", "T4", "T4_512")
METHODS = ("S2", "S3a_diff", "S3b_diff", "S3b_std")
SOURCE_FILES = {
    "src/models.py", "src/losses.py", "src/train.py", "src/evaluation.py",
    "src/dataset.py", "src/simulator.py", "src/spectral.py",
    "experiments/exp2_circular_trained.py", "experiments/exp2_reporting.py",
    "requirements-gpu.txt",
}


class AuditError(RuntimeError):
    pass


def require(condition, message):
    # Explicit checks remain active if Python is launched with -O.
    if not condition:
        raise AuditError(message)


def contained_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Invalid manifest path")
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root), f"Path escapes audited root: {relative}")
    require(candidate.is_file(), f"Missing file: {candidate}")
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_manifest(root: Path, manifest_path: Path):
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict) and bool(manifest), "Empty or invalid SHA256 manifest")
    normalized_names = set()
    for relative, digest in manifest.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest),
                f"Invalid SHA256 digest: {relative}")
        path = contained_path(root, relative)
        name = path.relative_to(root).as_posix()
        require(name not in normalized_names, f"Duplicate normalized manifest path: {relative}")
        normalized_names.add(name)
        require(sha256(path) == digest.lower(), f"Transfer SHA256 mismatch: {relative}")
    return manifest, normalized_names


def verify_task3_archive(repository: Path):
    provenance = repository / "reports/Task_3_1/provenance"
    hashes = load_json(provenance / "task3_archive_hashes.json")
    require(isinstance(hashes, dict) and len(hashes) == 42,
            "Task 3 archive manifest must contain the original 42 files")
    for relative, digest in hashes.items():
        require(relative.replace("\\", "/").startswith("reports/Task_3/"),
                f"Unexpected Task 3 archive path: {relative}")
        require(sha256(contained_path(repository, relative)) == digest,
                f"Archived Task 3 file changed: {relative}")
    previous = load_json(repository / "reports/Task_3/exp2_circular_results.json")
    previous_sources = previous["source_sha256"]
    require(set(previous_sources) == SOURCE_FILES and len(previous_sources) == 10,
            "Task 3 original JSON must identify the expected 10 source files")
    with zipfile.ZipFile(provenance / "task3_original_sources.zip") as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate filenames in preserved source archive")
        for relative, digest in previous_sources.items():
            require(relative in names, f"Missing preserved Task 3 source: {relative}")
            require(hashlib.sha256(archive.read(relative)).hexdigest() == digest,
                    f"Preserved Task 3 source disagrees with its result JSON: {relative}")
    return {"task3_archive_hashes_verified": len(hashes),
            "task3_original_source_hashes_verified": len(previous_sources)}


def unique_grid(rows, keys, expected, label):
    require(isinstance(rows, list), f"{label} must be a list")
    identities = [tuple(row[key] for key in keys) for row in rows]
    require(len(rows) == len(expected), f"{label}: {len(rows)} records, expected {len(expected)}")
    require(len(set(identities)) == len(identities), f"{label}: duplicate identities")
    require(set(identities) == expected, f"{label}: missing or unexpected identities")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def independently_recomputed_checks(report):
    """Recompute the fixed sixteen outcomes, allowing any scientific failures."""
    rows = [row for row in report["frequency_runs"]
            if row["split"] == "test_id_sub" and row["method"] == "S3b_diff"]
    results = [True]  # Completion/protocol is checked independently before this function.
    for model in ("M0", "M1", "M2"):
        selected = [row for row in rows if row["model"] == model and row["regime"] == "T1"]
        results.append(all(row["invalid"] == 0 and finite(row["slope_abs_error"])
                           and row["slope_abs_error"] < .01 and finite(row["r2"])
                           and row["r2"] > .9999 for row in selected))
    for regimes in (("T3",), ("T4", "T4_512")):
        for model in ("M0", "M1", "M2"):
            selected = [row for row in rows if row["model"] == model and row["regime"] in regimes]
            if model == "M0":
                result = all(row["invalid"] > 0 or not finite(row["r2"])
                             or (finite(row["slope_abs_error"]) and row["slope_abs_error"] > .3)
                             or row["r2"] < .80 for row in selected)
            else:
                result = all(row["invalid"] == 0 and finite(row["slope_abs_error"])
                             and row["slope_abs_error"] < .05 and finite(row["r2"])
                             and row["r2"] > .99 for row in selected)
            results.append(result)
    for regime in ("T1", "T3"):
        for model in ("M1", "M2", "M3"):
            selected = [row for row in report["scale_runs"] if row["split"] == "test_id_sub"
                        and row["model"] == model and row["regime"] == regime]
            if model == "M3":
                result = all(row["status"] == "completed" and finite(row.get("state_norm_r2"))
                             and finite(row.get("phase_norm_r2")) and row["state_norm_r2"] < .05
                             and row["phase_norm_r2"] < .05 for row in selected)
            else:
                result = all(row["status"] == "completed" and finite(row.get("phase_norm_r2"))
                             and finite(row.get("axis_probe_r2")) and row["phase_norm_r2"] > .90
                             and row["axis_probe_r2"] > .95 for row in selected)
            results.append(result)
    return results


def audit(incoming: Path, repository: Path):
    folder = incoming / "reports/Task_3_1"
    manifest_path = folder / "provenance/idh_artifact_sha256.json"
    manifest, manifest_names = verify_manifest(incoming, manifest_path)
    result_path = folder / "exp2_circular_results.json"
    report = load_json(result_path)
    require(report["status"] == "completed" and report["execution_completed"] is True,
            "Full experiment execution is incomplete")
    protocol = report["protocol"]
    require(protocol["smoke_test"] is False and protocol["seeds"] == list(SEEDS),
            "Expected production seeds 42..46, without smoke substitution")
    require(protocol["raw_test_samples"] == 2100, "Expected 2100 raw test samples")
    require(protocol["training_dtype"] == "float32" and protocol["evaluation_dtype"] == "float64",
            "Training/evaluation precision protocol differs")
    require(protocol["mlp_variant"] == "odd_residual" and protocol["d"] == 64
            and protocol["C"] == 4 and protocol["primary_method"] == "S3b_diff",
            "Unexpected encoder or primary estimator protocol")
    require(tuple(protocol["frequency_methods"]) == METHODS, "Frequency methods differ")
    config = protocol["training"]
    fixed = {"protocol": "task3_1", "steps": 2000, "batch_size": 256,
             "context_length": 16, "horizon": 64, "ar_steps": 16, "delta": 4,
             "temporal_lag": 16, "temporal_displacement_fraction": .8,
             "temporal_scale_weight": .2, "temporal_weight": 1.,
             "relative_denominator_epsilon": 1e-4, "orthogonality_weight": .2,
             "sigreg_weight": 1., "sparsity_weight": 0., "smoke_test": False}
    for key, expected in fixed.items():
        require(config[key] == expected, f"Unexpected training config {key}: {config[key]}")
    require(config["device"].startswith("cuda"), "Production training was not CUDA")
    metadata = protocol["regimes"]
    require(metadata["N_eval"] == metadata["H_extended"] == 512
            and metadata["H_train"] == 64 and metadata["macro_dt"] == .04
            and metadata["L"] == 16 and metadata["delta"] == 4 and metadata["dt"] == .01
            and metadata["origin_window_start_index"] == 0
            and metadata["excludes_z0"] is True and metadata["required_raw_samples"] == 2064,
            "Evaluation horizon/alignment metadata differs")
    for regime in REGIMES:
        samples = 64 if regime == "T4" else 512
        require(metadata["window_start_indices"][regime] == list(range(4, 4 * samples + 1, 4)),
                f"Incorrect window alignment: {regime}")
        require(abs(metadata["sample_span"][regime] - (samples - 1) * .04) < 1e-12,
                f"Incorrect sample-span metadata: {regime}")
    require(set(report["source_sha256"]) == SOURCE_FILES, "Expected exactly ten experiment source hashes")
    for relative, digest in report["source_sha256"].items():
        require(sha256(contained_path(repository, relative)) == digest,
                f"Deployed source differs from repository: {relative}")
    expected_training = {(seed, model) for seed in SEEDS for model in ("M1", "M2", "M3")}
    unique_grid(report["training_runs"], ("seed", "model"), expected_training, "training runs")
    required_transfers = {result_path.relative_to(incoming).as_posix()}
    for run in report["training_runs"]:
        identity = f"{run['model']}_seed{run['seed']}"
        require(run["status"] == "completed" and run["completed_steps"] == 2000
                and run["device"].startswith("cuda") and run["config"] == config,
                f"Incomplete or inconsistent GPU training run: {identity}")
        checkpoint_path = folder / "checkpoints" / f"{identity}.pt"
        required_transfers.add(checkpoint_path.relative_to(incoming).as_posix())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(checkpoint["step"] == 2000 and checkpoint["config"] == config,
                f"Checkpoint step/config mismatch: {identity}")
        provenance = checkpoint["provenance"]
        require(json.loads(json.dumps(provenance)) == run["provenance"],
                f"Checkpoint/report provenance mismatch: {identity}")
        require(provenance["seed"] == run["seed"] and provenance["objective_protocol"] == "task3_1"
                and provenance["device"].startswith("cuda") and provenance["dtype"] == "torch.float32"
                and "A100" in provenance["gpu_name"], f"Unexpected training provenance: {identity}")
        require(provenance["caller"]["source_sha256"] == report["source_sha256"],
                f"Checkpoint experiment source hashes differ: {identity}")
        for name, digest in provenance["source_sha256"].items():
            require(report["source_sha256"][f"src/{name}"] == digest,
                    f"Checkpoint training source mismatch: {identity}/{name}")
        architecture = provenance["model_config"]
        require(architecture["d"] == 64 and architecture["L"] == 16 and architecture["C"] == 4
                and architecture["mlp_variant"] == "odd_residual"
                and architecture["predictor_class"] == "ActionAngleKoopmanPredictor"
                and architecture["predictor_dt"] == .04,
                f"Unexpected checkpoint architecture: {identity}")
        history = checkpoint["history"]
        require(isinstance(history, list) and bool(history), f"Missing checkpoint history: {identity}")
        steps = [record["step"] for record in history]
        require(steps[0] == 1 and steps[-1] == 2000 and steps == sorted(set(steps))
                and history == run["history"], f"Checkpoint history mismatch: {identity}")
        require(bool(checkpoint["model_state_dict"]), f"Empty model checkpoint: {identity}")
        for key, tensor in checkpoint["model_state_dict"].items():
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu",
                    f"Invalid checkpoint tensor: {identity}/{key}")
            if tensor.is_floating_point():
                require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                        f"Nonfinite or wrong-precision checkpoint weight: {identity}/{key}")
        for field in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
            require(field in checkpoint, f"Missing resume state: {identity}/{field}")
        del checkpoint
    expected_frequencies = {(seed, model, split, regime, method) for seed in SEEDS
                            for model in MODELS for split in SPLITS for regime in REGIMES
                            for method in METHODS}
    expected_factors = {identity for identity in expected_frequencies if identity[1] == "M2"}
    for field, expected in (("frequency_runs", expected_frequencies),
                            ("factor_frequency_runs", expected_factors)):
        unique_grid(report[field], ("seed", "model", "split", "regime", "method"), expected, field)
        for row in report[field]:
            count = 30 if row["split"] == "test_id_sub" else 50
            samples = 64 if row["regime"] == "T4" else 512
            require(row["n_orbits"] == len(row["omega_est"]) == len(row["errors"])
                    == len(row["semi_major_axes"]) == count and row["n_samples"] == samples
                    and row["dt"] == .04 and row["projection"] == "centered_pca",
                    f"Frequency record dimensions/protocol differ: {field}/{row['seed']}/{row['model']}")
            require(row["semi_major_axes"] == report["test_grids"][row["split"]]
                    and all(finite(axis) and axis > 0 for axis in row["semi_major_axes"]),
                    "Frequency records use inconsistent radius grids")
            require(isinstance(row["invalid"], int) and 0 <= row["invalid"] <= count
                    and len(row["invalid_orbit_indices"]) == row["invalid"],
                    "Inconsistent invalid-orbit accounting")
            # Invalid frequencies and failed regressions are retained, never
            # rejected merely because the scientific criterion did not pass.
            if field == "factor_frequency_runs":
                require(len(row["active_factor_indices"]) == len(row["factor_temporal_variance_scores"]) == count
                        and all(isinstance(index, int) and 0 <= index < 4
                                for index in row["active_factor_indices"])
                        and all(len(scores) == 4 for scores in row["factor_temporal_variance_scores"]),
                        "Invalid leading-factor selection dimensions")
    expected_scales = {(seed, model, split, regime) for seed in SEEDS for model in MODELS
                       for split in SPLITS for regime in ("T1", "T3")}
    unique_grid(report["scale_runs"], ("seed", "model", "split", "regime"), expected_scales, "scale runs")
    for row in report["scale_runs"]:
        require(row["status"] in ("completed", "invalid"), "Unknown scale diagnostic status")
        for key in ("phase_norm_r2", "state_norm_r2", "axis_probe_r2", "scale_r2_cycle_radius",
                    "scale_r2_state_norm", "energy_r2"):
            require(key in row, f"Missing scale diagnostic: {key}")
        if row["status"] == "completed":
            count = 30 if row["split"] == "test_id_sub" else 50
            require(row["n_orbits"] == count and row["n_samples"] == 512 and row["latent_dim"] == 64,
                    "Scale diagnostic dimensions differ")
            for key in ("s", "r", "r_rms", "v_rms", "log_a", "physical_phase_norm"):
                require(len(row[key]) == count, f"Missing per-orbit scale values: {key}")
    expected_predictors = {(seed, model, split) for seed in SEEDS for model in MODELS for split in SPLITS}
    unique_grid(report["predictor_diagnostics"], ("seed", "model", "split"), expected_predictors,
                "predictor diagnostics")
    for row in report["predictor_diagnostics"]:
        count = 30 if row["split"] == "test_id_sub" else 50
        require(row["structured_predictor"] is (row["model"] != "M0") and row["temporal_lag"] == 16,
                "Predictor diagnostic architecture/lag differs")
        require(len(row["input_temporal_rms"]) == len(row["latent_temporal_rms"]) == count,
                "Predictor temporal diagnostic dimensions differ")
        for key in ("norm_relative_drift_max", "ar_direct_relative_error_max", "action_log_drift_max",
                    "orthogonality_max_abs", "temporal_displacement_ratio_median"):
            require(key in row, f"Missing predictor diagnostic: {key}")
    unique_grid(report["portraits"], ("seed", "model", "split"),
                {(42, model, "test_id_sub") for model in ("M0", "M1", "M2")}, "portraits")
    checks = report["checks"]
    require(len(checks) == 16 and len({check["criterion"] for check in checks}) == 16
            and all(isinstance(check["passed"], bool) for check in checks),
            "Expected sixteen unique boolean scientific criteria")
    recomputed = independently_recomputed_checks(report)
    require([check["passed"] for check in checks] == recomputed,
            "Reported scientific outcomes disagree with independent threshold recomputation")
    require(report["acceptance_passed"] is all(recomputed),
            "Scientific acceptance summary disagrees with individual outcomes")
    unit_candidates = sorted(folder.glob("*gpu*unit*test*.txt"))
    require(len(unit_candidates) == 1, "Expected exactly one GPU unit-test output in Task_3_1")
    unit_path = unit_candidates[0]
    unit_text = unit_path.read_text(encoding="utf-8-sig")
    require(re.search(r"\bRan 134 tests\b", unit_text) and unit_text.rstrip().endswith("OK"),
            "GPU unit-test output does not confirm 134 passing tests")
    required_transfers.add(unit_path.relative_to(incoming).as_posix())
    figure_path = folder / "figures/exp2_circular_regimes.png"
    required_transfers.add(figure_path.relative_to(incoming).as_posix())
    with Image.open(figure_path) as figure:
        size, dpi = figure.size, figure.info.get("dpi")
        require(size == (5400, 1860), f"Publication figure dimensions differ: {size}")
        require(dpi is not None and len(dpi) == 2 and all(abs(value - 300) < .01 for value in dpi),
                f"Publication figure is not 300 dpi: {dpi}")
    for relative in ("reports/Task_3_1/exp2_circular_report.md",
                     "reports/Task_3_1/exp2_circular_stdout.txt"):
        require(contained_path(incoming, relative).stat().st_size > 0, f"Empty required report: {relative}")
        required_transfers.add(relative)
    require(required_transfers <= manifest_names,
            f"Required files missing from transfer manifest: {sorted(required_transfers - manifest_names)}")
    archive_summary = verify_task3_archive(repository)
    return {"transfer_and_completeness_passed": True,
            "audit_scope": "Artifact integrity, execution completeness and honest criterion accounting; scientific success is not required.",
            "artifact_hashes_verified": len(manifest), "source_hashes_verified": len(report["source_sha256"]),
            "checkpoints_verified_at_step_2000": len(report["training_runs"]),
            "checkpoint_loading": "weights_only=True, map_location=cpu", "gpu_unit_tests_passed": 134,
            "raw_test_samples": 2100, "frequency_records": len(report["frequency_runs"]),
            "factor_frequency_records": len(report["factor_frequency_runs"]),
            "per_orbit_frequency_estimates": sum(row["n_orbits"] for row in report["frequency_runs"]),
            "per_orbit_factor_estimates": sum(row["n_orbits"] for row in report["factor_frequency_runs"]),
            "scale_records": len(report["scale_runs"]), "predictor_diagnostic_records": len(report["predictor_diagnostics"]),
            "figure_pixels": list(size), "figure_dpi": list(dpi), **archive_summary,
            "scientific_outcomes_independently_recomputed": True,
            "scientific_acceptance_passed": report["acceptance_passed"],
            "scientific_criteria_passed": sum(recomputed), "scientific_criteria_total": len(checks),
            "failed_scientific_criteria": [check["criterion"] for check in checks if not check["passed"]],
            "environment": report["environment"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incoming", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.incoming.resolve(), args.repository.resolve())
        exit_code = 0
    except Exception as error:
        result = {"transfer_and_completeness_passed": False, "error_type": type(error).__name__,
                  "error": str(error), "scientific_acceptance_claimed": False}
        exit_code = 1
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.resolve().write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
