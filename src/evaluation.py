"""Task 3/3.1 evaluation without orbital metadata in frequency estimators.

T1/T2/T3 exclude the initial state and align to window starts delta..N_eval*delta.
T4 evaluates only the trained, direct horizons 1..H_train from the same origin.
The legacy default uses independently fitted, centered and unwhitened PCA.
Task 3.1 may explicitly choose uncentered orthogonal SVD projection: it preserves
the encoder's supplied origin, without claiming that a finite arc has zero mean.

Scale scores are in-sample linear diagnostics, not information-theoretic tests.
In particular, low linear R2 does not establish absence of nonlinear information,
and a constant state norm does not imply a constant trajectory cycle radius.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import torch

from .spectral import (evaluate_kepler_regression, freq_s2_interpolated_fft,
                       freq_s3a_dmd_koopman, freq_s3b_phase_unwrap,
                       project_to_2d_mode)

METHODS = ("S2", "S3a_diff", "S3b_diff", "S3b_std")


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive_scalar(value, name):
    if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _axes(values, count):
    if np.iscomplexobj(values):
        raise ValueError("semi_major_axes must be real")
    axes = np.asarray(values, dtype=np.float64)
    if axes.shape != (count,) or not np.all(np.isfinite(axes)) or np.any(axes <= 0):
        raise ValueError("semi_major_axes must have finite positive shape (N_orbits,)")
    return axes


def _latent(values, *, finite=True):
    if np.iscomplexobj(values):
        raise ValueError("latent trajectories must be real")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 2 or array.shape[1] < 2 or array.shape[2] < 1:
        raise ValueError("latent trajectories must have shape (N>=2,T>=2,d>=1)")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError("latent trajectories must be finite")
    return array


def _linear_fit(features, target, names, tolerance):
    """Fit centered, scaled OLS while dropping numerically constant predictors.

    A float32-normalized sphere has tiny apparent norm variation after promotion
    to float64. A predictor with log-coordinate std <= 1e-6 by default is treated
    as constant. The threshold, raw stds and dropped features are returned, so
    small-scale variation is disclosed rather than counted as reliable scale.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    feature_std = np.sqrt(np.mean(centered * centered, axis=0))
    active = feature_std > tolerance
    target_mean = float(target.mean())
    centered_target = target - target_mean
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    rank = 0
    if np.any(active):
        standardized = centered[:, active] / feature_std[active]
        scaled_coef, _, rank, _ = np.linalg.lstsq(standardized, centered_target, rcond=None)
        coefficients[active] = scaled_coef / feature_std[active]
    fitted_centered = centered @ coefficients
    residual = centered_target - fitted_centered
    sse, sst = float(residual @ residual), float(centered_target @ centered_target)
    defined = not np.all(target == target[0])
    ratio = sse / sst if defined and sst > 0 else float("nan")
    return {"r2": 1.0 - ratio, "one_minus_r2": ratio, "sse": sse, "sst": sst,
            "intercept": float(target_mean - feature_mean @ coefficients),
            "coefficients": {name: float(value) for name, value in zip(names, coefficients)},
            "feature_std": {name: float(value) for name, value in zip(names, feature_std)},
            "active_features": [name for name, enabled in zip(names, active) if enabled],
            "dropped_constant_features": [name for name, enabled in zip(names, active) if not enabled],
            "feature_std_tolerance": tolerance, "design_rank": int(rank),
            "status": "ok" if defined else "undefined_constant_response"}


def _rank(values):
    """Average ranks for ties without adding a scipy runtime dependency."""
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranked[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return ranked


def _spearman(x, y, tolerance):
    if np.std(x) <= tolerance or np.all(y == y[0]):
        return float("nan")
    xr, yr = _rank(x), _rank(y)
    xr, yr = xr - xr.mean(), yr - yr.mean()
    denominator = np.sqrt((xr @ xr) * (yr @ yr))
    return float((xr @ yr) / denominator) if denominator else float("nan")


def evaluate_scale_preservation(Z_all_orbits, semi_major_axes, mu=1.0, *,
                                numerical_feature_tol=1e-6, dt_pred=0.04):
    """Evaluate the specified scale/energy regressions and retain all predictors.

    s_i = log(sqrt(mean_t(||z_it - mean_t(z_i)||^2) + 1e-12)); r_i
    uses the uncentered norm. Energy is log(mu/(2*a)), so its target contains no
    independent information beyond log(a). A constant predictor against varying
    axes gives R2=0. Constant axes give undefined R2, never a spurious perfect fit.
    Legacy s/r use coordinate-summed norms. Task 3.1 also reports coordinate-RMS
    r_rms/v_rms and probes (a dimension-dependent additive shift leaves the old
    R2 values unchanged). phase_norm_r2 regresses r_rms *as the response* on the
    true physical phase norm, not the reverse regression used by state_norm_r2.
    A numerically constant response has undefined conventional R2; Task 3.1's
    requested zero-score convention is explicit in its returned fit/flag.
    All input states must be finite; callers must retain failed-run diagnostics.
    """
    values = _latent(Z_all_orbits)
    axes = _axes(semi_major_axes, len(values))
    mu = _positive_scalar(mu, "mu")
    tolerance = _positive_scalar(numerical_feature_tol, "numerical_feature_tol")
    dt_pred = _positive_scalar(dt_pred, "dt_pred")
    centered = values - values.mean(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        cycle_square = np.mean(np.sum(centered * centered, axis=2), axis=1)
        state_square = np.mean(np.sum(values * values, axis=2), axis=1)
        s, r = 0.5 * np.log(cycle_square + 1e-12), 0.5 * np.log(state_square + 1e-12)
        state_rms_square = np.mean(values * values, axis=(1, 2))
        velocity_rms_square = np.mean((np.diff(values, axis=1) / dt_pred) ** 2,
                                      axis=(1, 2))
        r_rms = 0.5 * np.log(state_rms_square + 1e-12)
        v_rms = 0.5 * np.log(velocity_rms_square + 1e-12)
    if not all(np.all(np.isfinite(value)) for value in (s, r, r_rms, v_rms)):
        raise ValueError("latent norms overflowed; scale regressions are undefined")
    log_a = np.log(axes)
    log_energy = np.log(mu / 2) - log_a
    cycle_fit = _linear_fit(s, log_a, ["s"], tolerance)
    state_fit = _linear_fit(r, log_a, ["r"], tolerance)
    energy_fit = _linear_fit(np.column_stack((s, r)), log_energy, ["s", "r"], tolerance)
    # Any constant factor in physical RMS changes only the intercept. The
    # specified feature omits that factor; actual mean over four raw channels
    # is (a**2 + mu/a)/4, rather than the task prose's factor 1/2.
    physical_phase_norm = 0.5 * np.log(axes**2 + mu / axes)
    phase_fit = _linear_fit(physical_phase_norm, r_rms, ["physical_phase_norm"], tolerance)
    response_std = float(np.std(r_rms))
    constant_norm = response_std <= tolerance
    phase_fit.update(response_std=response_std, response_std_tolerance=tolerance,
                     constant_response_convention_applied=constant_norm)
    if constant_norm:
        phase_fit.update(conventional_r2=phase_fit["r2"], r2=0.0, one_minus_r2=1.0,
                         status="numerically_constant_response_zero_score_convention")
    axis_fit = _linear_fit(np.column_stack((r_rms, v_rms)), log_a,
                           ["r_rms", "v_rms"], tolerance)
    physical_state_fit = _linear_fit(physical_phase_norm, log_a,
                                     ["physical_phase_norm"], tolerance)
    rho = _spearman(s, log_a, tolerance)
    return {"scale_r2_cycle_radius": cycle_fit["r2"],
            "scale_r2_state_norm": state_fit["r2"], "energy_r2": energy_fit["r2"],
            "state_norm_r2": state_fit["r2"], "phase_norm_r2": phase_fit["r2"],
            "axis_probe_r2": axis_fit["r2"],
            "state_norm_numerically_constant": constant_norm,
            "physical_state_norm_axis_r2": physical_state_fit["r2"],
            "n_orbits": len(values), "n_samples": values.shape[1], "latent_dim": values.shape[2],
            "s": s.tolist(), "r": r.tolist(), "log_a": log_a.tolist(),
            "r_rms": r_rms.tolist(), "v_rms": v_rms.tolist(), "dt_pred": dt_pred,
            "physical_phase_norm": physical_phase_norm.tolist(),
            "log_abs_energy": log_energy.tolist(),
            "cycle_radius_rms": np.sqrt(cycle_square + 1e-12).tolist(),
            "state_norm_rms": np.sqrt(state_square + 1e-12).tolist(),
            "fits": {"cycle_radius": cycle_fit, "state_norm": state_fit, "energy": energy_fit,
                     "phase_norm": phase_fit, "axis_probe": axis_fit,
                     "physical_state_norm_axis": physical_state_fit},
            "cycle_radius_spearman": rho,
            "cycle_radius_direction": ("increasing" if rho > 0 else "decreasing" if rho < 0
                                       else "constant_or_undefined" if not np.isfinite(rho)
                                       else "nonmonotonic"),
            "energy_target_is_affine_log_a": True,
            "rms_coordinate_convention": "r/s are legacy coordinate-summed; r_rms/v_rms average coordinates",
            "physical_phase_norm_convention": "0.5*log(a**2+mu/a); additive RMS constants do not change R2",
            "phase_norm_constant_response_convention": "R2=0 when std(r_rms)<=numerical_feature_tol; conventional R2 is undefined for an exactly constant response",
            "interpretation": "In-sample linear probes; low R2 does not establish loss of all scale information."}


def evaluate_oos_scale_probes(train_scale, test_scale, *, tolerance=1e-6):
    """Fit only on train orbits, apply frozen coefficients to a test split.

    Uses the same declared features/directions as the in-sample probes.
    Constant columns are selected using train only. Negative test R2 is kept;
    a numerically constant phase response uses the disclosed zero convention.
    """
    result = {"status": "completed", "fit_split": "train", "n_fit_orbits": train_scale["n_orbits"],
              "n_test_orbits": test_scale["n_orbits"], "fits": {}}
    specifications = (("phase_norm", ["physical_phase_norm"], "r_rms"),
                      ("axis_probe", ["r_rms", "v_rms"], "log_a"))
    for name, features, response in specifications:
        x_train = np.column_stack([train_scale[key] for key in features])
        y_train = np.asarray(train_scale[response], dtype=np.float64)
        x_test = np.column_stack([test_scale[key] for key in features])
        y_test = np.asarray(test_scale[response], dtype=np.float64)
        if not all(np.isfinite(v).all() for v in (x_train, y_train, x_test, y_test)):
            raise ValueError("All train/test probe features must be finite; no orbit filtering")
        fit = _linear_fit(x_train, y_train, features, tolerance)
        prediction = fit["intercept"] + x_test @ np.array([fit["coefficients"][f] for f in features])
        residual = y_test - prediction
        centered = y_test - y_test.mean()
        sse, sst = float(residual @ residual), float(centered @ centered)
        conventional = 1 - sse / sst if sst > 0 else float("nan")
        constant = name == "phase_norm" and float(y_test.std()) <= tolerance
        score = 0.0 if constant else conventional
        result[f"oos_{name}_r2"] = score
        result["fits"][name] = {"train_fit": fit, "test_sse": sse, "test_sst": sst,
                                "test_predictions": prediction.tolist(), "test_targets": y_test.tolist(),
                                "test_r2": score, "conventional_test_r2": conventional,
                                "constant_response_convention_applied": constant,
                                "response_std_tolerance": tolerance}
    return result


def regime_metadata(*, N_eval=256, L=16, delta=4, H_train=64, dt=0.01, start=0,
                    H_extended=None):
    """Return exact window-start indices and time coordinates for each regime."""
    N_eval, L, delta, H_train, start = (
        _integer(value, name, minimum=minimum) for value, name, minimum in (
            (N_eval, "N_eval", 2), (L, "L", 1), (delta, "delta", 1),
            (H_train, "H_train", 2), (start, "start", 0)))
    dt = _positive_scalar(dt, "dt")
    indices = {name: (start + delta * np.arange(1, count + 1)).tolist()
               for name, count in (("T1", N_eval), ("T2", N_eval), ("T3", N_eval), ("T4", H_train))}
    extended_name = None
    if H_extended is not None:
        H_extended = _integer(H_extended, "H_extended", minimum=H_train + 1)
        extended_name = f"T4_{H_extended}"
        indices[extended_name] = (start + delta * np.arange(1, H_extended + 1)).tolist()
    return {"N_eval": N_eval, "L": L, "delta": delta, "H_train": H_train,
            "dt": dt, "macro_dt": delta * dt, "origin_window_start_index": start,
            "excludes_z0": True, "window_start_indices": indices,
            "H_extended": H_extended, "extended_direct_regime": extended_name,
            "sample_times": {name: (np.asarray(index, dtype=np.float64) * dt).tolist()
                             for name, index in indices.items()},
            "required_raw_samples": start + N_eval * delta + L,
            "sample_span": {name: (len(index) - 1) * delta * dt for name, index in indices.items()},
            "last_horizon_from_origin": {name: len(index) * delta * dt for name, index in indices.items()},
            "pca": "frequency evaluation chooses centered_pca (legacy default) or uncentered_svd explicitly"}


def evaluate_regimes(model, trajectories, *, N_eval=256, L=16, delta=4,
                     H_train=64, dt=0.01, start=0, device=None, dtype=None,
                     batch_size=1024, H_extended=None):
    """Return T1..T4 NumPy arrays (N,T,d), sharing an initial encoded window.

    T1 encodes true future windows; T2 predicts each from the true immediately
    preceding window. T3 uses only the initial window then repeatedly token 1.
    T4 uses only that initial window, independently with horizons 1..H_train.
    Optional H_extended returns a separate T4_{H_extended}, never replacing T4.
    Horizons beyond a learned token table require supports_extrapolation=True.
    Generated nonfinite states are retained for honest failed-orbit evaluation.
    The model's train/eval mode is restored. Model weights are never converted.
    """
    metadata = regime_metadata(N_eval=N_eval, L=L, delta=delta, H_train=H_train,
                               dt=dt, start=start, H_extended=H_extended)
    batch_size = _integer(batch_size, "batch_size")
    if np.iscomplexobj(trajectories):
        raise ValueError("trajectories must be real")
    states = np.asarray(trajectories)
    encoder = model.context_encoder
    if (states.ndim != 3 or states.shape[0] < 1 or states.shape[-1] != encoder.C
            or states.shape[1] < metadata["required_raw_samples"]
            or not np.all(np.isfinite(states))):
        raise ValueError("trajectories must be finite (N,T,C) with enough raw samples and matching channels")
    if encoder.L != L:
        raise ValueError("L must match the model encoder")
    maximum_direct = H_train if H_extended is None else H_extended
    if (model.predictor.H_max < maximum_direct
            and not getattr(model.predictor, "supports_extrapolation", False)):
        raise ValueError("T4 exceeds the model's available horizon tokens")
    first_parameter = next(model.parameters(), None)
    inferred_device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    inferred_dtype = first_parameter.dtype if first_parameter is not None else torch.float32
    device = inferred_device if device is None else torch.device(device)
    dtype = inferred_dtype if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    same_device = (device.type == inferred_device.type and
                   (device.index is None or device.index == inferred_device.index))
    if first_parameter is not None and (not same_device or dtype != inferred_dtype):
        raise ValueError("device/dtype must match model weights; evaluation must not convert trained weights")
    window_starts = start + delta * np.arange(N_eval + 1)
    sample_indices = window_starts[:, None] + np.arange(L)[None, :]
    windows = torch.as_tensor(np.array(states[:, sample_indices, :], copy=True), device=device, dtype=dtype)
    original_mode = model.training
    try:
        model.eval()
        with torch.inference_mode():
            # Matrix exponentials are constant throughout evaluation. Reusing
            # one Q also prevents hundreds of numerically distinct recomputes.
            basis = (model.predictor.orthogonal_basis()
                     if hasattr(model.predictor, "orthogonal_basis") else None)

            def predict(z, horizon):
                if basis is not None:
                    return model.predictor(z, horizon, basis=basis)
                return model.predictor(z, horizon)

            encoded = model.encode_sequence(windows, batch_size=batch_size)
            z0 = encoded[:, 0]
            predecessors = encoded[:, :-1].reshape(-1, encoded.shape[-1])
            teacher = torch.cat([predict(chunk, 1) for chunk in predecessors.split(batch_size)])
            teacher = teacher.reshape(states.shape[0], N_eval, encoded.shape[-1])
            state, autoregressive = z0, []
            for _ in range(N_eval):
                state = predict(state, 1)
                autoregressive.append(state)
            if hasattr(model.predictor, "predict_horizons"):
                horizons = torch.arange(1, maximum_direct + 1, device=device)
                direct_all = model.predictor.predict_horizons(z0, horizons, basis=basis)
            else:
                direct_all = torch.stack([predict(z0, k)
                                          for k in range(1, maximum_direct + 1)], dim=1)
            sequences = {"T1": encoded[:, 1:], "T2": teacher,
                         "T3": torch.stack(autoregressive, dim=1),
                         "T4": direct_all[:, :H_train]}
            if H_extended is not None:
                sequences[metadata["extended_direct_regime"]] = direct_all
            return {key: value.detach().cpu().to(torch.float64).numpy().copy()
                    for key, value in sequences.items()}
    finally:
        model.train(original_mode)


def evaluate_frequencies(Z_all_orbits, semi_major_axes, dt=0.04, *,
                         projection="centered_pca"):
    """Return all four methods, retaining every failed orbit in the regression.

    Each record includes omega_est, errors, invalid count, and Kepler metrics.
    One invalid/nonpositive frequency makes the full regression undefined: no
    selection of successful orbits is made. A constant-frequency fit retains
    the existing spectral module's undefined R2 semantics.
    """
    values = _latent(Z_all_orbits, finite=False)
    if values.shape[-1] < 2:
        raise ValueError("Frequency evaluation needs latent_dim >= 2")
    axes = _axes(semi_major_axes, len(values))
    dt = _positive_scalar(dt, "dt")
    if projection not in ("centered_pca", "uncentered_svd"):
        raise ValueError("projection must be centered_pca or uncentered_svd")
    estimates = {method: [] for method in METHODS}
    errors = {method: [] for method in METHODS}
    for orbit in values:
        try:
            if projection == "centered_pca":
                mode = project_to_2d_mode(orbit)
            else:
                if not np.all(np.isfinite(orbit)):
                    raise ValueError("Trajectory must contain only finite values.")
                # Orthogonal scores preserve the supplied origin. An odd map
                # has zero full-cycle mean, not zero mean on every finite arc.
                if orbit.shape[1] == 2:
                    mode = orbit.copy()
                else:
                    _, _, right = np.linalg.svd(orbit, full_matrices=False)
                    mode = orbit @ right[:2].T
            projection_error = None
        except (ValueError, np.linalg.LinAlgError) as exc:
            mode, projection_error = None, str(exc)
        for method in METHODS:
            try:
                if projection_error:
                    raise ValueError(projection_error)
                if method == "S2":
                    frequency = freq_s2_interpolated_fft(mode, dt)
                elif method == "S3a_diff":
                    frequency = freq_s3a_dmd_koopman(mode, dt, use_differences=True)
                else:
                    frequency = freq_s3b_phase_unwrap(mode, dt, use_differences=method.endswith("diff"))
                if not np.isfinite(frequency) or frequency <= 0:
                    raise ValueError("Frequency estimate is nonfinite or nonpositive")
                error = None
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                frequency, error = float("nan"), str(exc)
            estimates[method].append(float(frequency))
            errors[method].append(error)
    results = {}
    for method in METHODS:
        frequency = np.asarray(estimates[method])
        invalid = int(np.count_nonzero(~np.isfinite(frequency) | (frequency <= 0)))
        metrics = {name: float("nan") for name in ("slope", "intercept", "slope_abs_error", "r2",
                                                  "one_minus_r2", "log10_one_minus_r2")}
        regression_error = None
        if invalid == 0:
            try:
                metrics = evaluate_kepler_regression(axes, frequency)
            except ValueError as exc:
                regression_error = str(exc)
        else:
            regression_error = "At least one invalid orbit; regression is undefined without filtering."
        results[method] = {"method": method, "n_orbits": len(values), "n_samples": values.shape[1],
                           "dt": dt, "semi_major_axes": axes.tolist(), "omega_est": estimates[method],
                           "projection": projection,
                           "errors": errors[method], "invalid": invalid,
                           "invalid_orbit_indices": np.flatnonzero(~np.isfinite(frequency) | (frequency <= 0)).tolist(),
                           "regression_error": regression_error, **metrics}
    return results
