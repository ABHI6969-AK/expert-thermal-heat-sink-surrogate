#!/usr/bin/env python3
"""Heat-sink parameter sweep and machine-learning surrogate assessment.

The heat-transfer correlations are simplified screening equations adapted from
the supplied assessment materials. They are not a substitute for CFD or a
validated thermal design tool. Run with:

    python thermal_assessment.py

Dependencies: numpy and pandas.
Outputs are written beside this script unless --output-dir is supplied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThermalConfig:
    """Fixed geometry and material properties for the reference heat sink."""

    die_length_m: float = 0.0525
    die_width_m: float = 0.045
    sink_length_m: float = 0.090
    sink_width_m: float = 0.116
    sink_height_m: float = 0.027
    fin_thickness_m: float = 0.0008
    fin_count: int = 60
    base_thickness_m: float = 0.0025
    aluminum_conductivity_W_mK: float = 167.0
    tim_thickness_m: float = 0.0001
    junction_to_case_C_per_W: float = 0.2
    ambient_C: float = 40.0
    # Air properties are retained from the supplied assessment's 25 C values.
    # For higher accuracy, evaluate properties at the air film temperature.
    air_conductivity_W_mK: float = 0.0262
    air_kinematic_viscosity_m2_s: float = 1.57e-5
    air_prandtl: float = 0.71
    reynolds_transition: float = 2300.0


CONFIG = ThermalConfig()
INPUT_COLUMNS = ["TDP_W", "air_velocity_m_s", "k_tim_W_mK"]
TARGET_COLUMNS = ["R_total_C_per_W", "T_junction_C"]
RANDOM_SEED = 20261105


def validate_config(config: ThermalConfig) -> None:
    """Reject invalid fixed geometry before producing any results."""
    positive = {
        "die_length_m": config.die_length_m,
        "die_width_m": config.die_width_m,
        "sink_length_m": config.sink_length_m,
        "sink_width_m": config.sink_width_m,
        "sink_height_m": config.sink_height_m,
        "fin_thickness_m": config.fin_thickness_m,
        "fin_count": config.fin_count,
        "base_thickness_m": config.base_thickness_m,
        "aluminum_conductivity_W_mK": config.aluminum_conductivity_W_mK,
        "tim_thickness_m": config.tim_thickness_m,
        "air_conductivity_W_mK": config.air_conductivity_W_mK,
        "air_kinematic_viscosity_m2_s": config.air_kinematic_viscosity_m2_s,
        "air_prandtl": config.air_prandtl,
    }
    invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"These configuration values must be finite and positive: {invalid}")
    if config.sink_height_m <= config.base_thickness_m:
        raise ValueError("Sink height must exceed base thickness.")
    if config.sink_width_m <= config.fin_count * config.fin_thickness_m:
        raise ValueError("Fin thicknesses exceed the available sink width.")
    if config.fin_count < 2:
        raise ValueError("At least two fins are required by the spacing formula.")
    if config.junction_to_case_C_per_W < 0:
        raise ValueError("Junction-to-case resistance cannot be negative.")


def heat_sink_model(
    tdp_W: float | np.ndarray,
    air_velocity_m_s: float | np.ndarray,
    k_tim_W_mK: float | np.ndarray,
    config: ThermalConfig = CONFIG,
) -> dict[str, np.ndarray]:
    """Calculate thermal resistance and junction temperature for broadcastable inputs.

    The model uses the source assessment's laminar developing-flow correlation
    below Re=2300 and Dittus-Boelter above it. A straight-fin efficiency factor
    accounts approximately for conduction along aluminum fins. Base conduction,
    TIM resistance, and junction-to-case resistance are included in R_total.
    """
    validate_config(config)
    tdp, velocity, k_tim = np.broadcast_arrays(
        np.asarray(tdp_W, dtype=float),
        np.asarray(air_velocity_m_s, dtype=float),
        np.asarray(k_tim_W_mK, dtype=float),
    )
    for name, values in (
        ("TDP_W", tdp),
        ("air_velocity_m_s", velocity),
        ("k_tim_W_mK", k_tim),
    ):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} values must be finite and greater than zero.")

    cfg = config
    die_area = cfg.die_length_m * cfg.die_width_m
    fin_spacing = (
        cfg.sink_width_m - cfg.fin_count * cfg.fin_thickness_m
    ) / (cfg.fin_count - 1)
    if fin_spacing <= 0:
        raise ValueError("Calculated fin spacing must be positive.")

    reynolds = velocity * fin_spacing / cfg.air_kinematic_viscosity_m2_s
    nusselt_laminar = 1.86 * (
        reynolds * cfg.air_prandtl * (2.0 * fin_spacing) / cfg.sink_length_m
    ) ** (1.0 / 3.0)
    nusselt_turbulent = (
        0.023 * reynolds**0.8 * cfg.air_prandtl**0.3
    )
    nusselt = np.where(
        reynolds < cfg.reynolds_transition,
        nusselt_laminar,
        nusselt_turbulent,
    )
    h = nusselt * cfg.air_conductivity_W_mK / (2.0 * fin_spacing)

    fin_height = cfg.sink_height_m - cfg.base_thickness_m
    # Corrected fin length approximates convection from the fin tip.
    corrected_fin_length = fin_height + cfg.fin_thickness_m / 2.0
    fin_cross_section = cfg.fin_thickness_m * cfg.sink_length_m
    fin_perimeter = 2.0 * (cfg.sink_length_m + cfg.fin_thickness_m)
    fin_m = np.sqrt(
        h * fin_perimeter
        / (cfg.aluminum_conductivity_W_mK * fin_cross_section)
    )
    fin_mL = fin_m * corrected_fin_length
    fin_efficiency = np.tanh(fin_mL) / fin_mL

    fin_area_corrected = (
        cfg.fin_count * 2.0 * cfg.sink_length_m * corrected_fin_length
    )
    exposed_base_area = (
        cfg.sink_length_m
        * (cfg.sink_width_m - cfg.fin_count * cfg.fin_thickness_m)
    )
    effective_area = fin_efficiency * fin_area_corrected + exposed_base_area

    r_conv = 1.0 / (h * effective_area)
    r_tim = cfg.tim_thickness_m / (k_tim * die_area)
    r_cond = cfg.base_thickness_m / (
        cfg.aluminum_conductivity_W_mK * die_area
    )
    r_total = cfg.junction_to_case_C_per_W + r_tim + r_cond + r_conv
    t_junction = cfg.ambient_C + tdp * r_total

    return {
        "Re": reynolds,
        "Nu": nusselt,
        "h_W_m2K": h,
        "fin_spacing_m": np.full_like(reynolds, fin_spacing),
        "fin_efficiency": fin_efficiency,
        "effective_area_m2": effective_area,
        "R_conv_C_per_W": r_conv,
        "R_TIM_C_per_W": r_tim,
        "R_cond_C_per_W": np.full_like(r_conv, r_cond),
        "R_total_C_per_W": r_total,
        "T_junction_C": t_junction,
    }


def make_dataset() -> pd.DataFrame:
    """Create a deterministic 3-D grid with 19,375 cases across requested ranges."""
    tdp_values = np.linspace(30.0, 250.0, 25)
    velocity_values = np.linspace(0.5, 15.0, 31)
    k_tim_values = np.linspace(1.0, 12.0, 25)
    tdp, velocity, k_tim = np.meshgrid(
        tdp_values, velocity_values, k_tim_values, indexing="ij"
    )
    flat_tdp, flat_velocity, flat_k_tim = (
        tdp.ravel(), velocity.ravel(), k_tim.ravel()
    )
    outputs = heat_sink_model(flat_tdp, flat_velocity, flat_k_tim)
    return pd.DataFrame(
        {
            "TDP_W": flat_tdp,
            "air_velocity_m_s": flat_velocity,
            "k_tim_W_mK": flat_k_tim,
            **outputs,
        }
    )


def ridge_features(x: np.ndarray) -> np.ndarray:
    """Create physics-motivated features for an interpretable Ridge baseline.

    In this flow range, the supplied laminar correlation makes convection
    approximately proportional to velocity^(-1/3); TIM resistance scales as
    1/k_TIM; and T_junction = T_ambient + TDP * R_total.
    """
    tdp, velocity, k_tim = x.T
    v_m13 = velocity ** (-1.0 / 3.0)
    v_m16 = velocity ** (-1.0 / 6.0)
    inv_k = 1.0 / k_tim
    return np.column_stack(
        [
            np.ones_like(tdp),
            tdp,
            velocity,
            k_tim,
            v_m13,
            v_m16,
            inv_k,
            tdp * v_m13,
            tdp * v_m16,
            tdp * inv_k,
        ]
    )


def fit_ridge(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1e-6
) -> dict[str, np.ndarray]:
    """Fit a multi-output Ridge model using a centered, scaled feature basis."""
    feature_train = ridge_features(x_train)
    intercept = feature_train[:, :1]
    nonconstant = feature_train[:, 1:]
    mean = nonconstant.mean(axis=0)
    scale = nonconstant.std(axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack([intercept, (nonconstant - mean) / scale])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Predict both targets with the fitted physics-feature Ridge model."""
    features = ridge_features(x)
    design = np.column_stack(
        [features[:, :1], (features[:, 1:] - model["mean"]) / model["scale"]]
    )
    return design @ model["coefficients"]


def predict_weighted_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 12,
    chunk_size: int = 128,
) -> np.ndarray:
    """Predict with distance-weighted KNN using standardized inputs."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    train_scaled = (x_train - mean) / scale
    query_scaled = (x_query - mean) / scale
    predictions: list[np.ndarray] = []
    for start in range(0, len(query_scaled), chunk_size):
        query = query_scaled[start : start + chunk_size]
        distances_squared = np.sum(
            (query[:, None, :] - train_scaled[None, :, :]) ** 2, axis=2
        )
        nearest = np.argpartition(distances_squared, k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances_squared, nearest, axis=1)
        weights = 1.0 / np.maximum(np.sqrt(nearest_distances), 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        predictions.append(np.sum(y_train[nearest] * weights[:, :, None], axis=1))
    return np.vstack(predictions)


def evaluate_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit Ridge and KNN surrogates; report metrics by target on held-out levels."""
    x = data[INPUT_COLUMNS].to_numpy(dtype=float)
    y = data[TARGET_COLUMNS].to_numpy(dtype=float)

    # Hold out complete velocity levels, not random rows, to reduce leakage.
    velocity_levels = data["air_velocity_m_s"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    test_level_count = max(1, int(np.ceil(0.20 * len(velocity_levels))))
    test_levels = np.sort(
        rng.choice(velocity_levels, size=test_level_count, replace=False)
    )
    test_mask = data["air_velocity_m_s"].isin(test_levels).to_numpy()
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ridge_model = fit_ridge(x_train, y_train)
    predictions_by_model = {
        "Physics-feature Ridge": predict_ridge(ridge_model, x_test),
        "Distance-weighted KNN (k=12)": predict_weighted_knn(
            x_train, y_train, x_test, k=12
        ),
    }
    rows: list[dict[str, float | str]] = []
    for model_name, predictions in predictions_by_model.items():
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            truth = y_test[:, target_index]
            pred = predictions[:, target_index]
            residual = truth - pred
            rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "MAE": float(np.mean(np.abs(residual))),
                    "RMSE": float(np.sqrt(np.mean(residual**2))),
                    "R2": float(
                        1.0 - np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2)
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    split_details = {
        "method": "Hold out complete air-velocity levels",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_velocity_levels_m_s": sorted(
            data.iloc[train_idx]["air_velocity_m_s"].unique().astype(float).tolist()
        ),
        "test_velocity_levels_m_s": test_levels.astype(float).tolist(),
        "random_seed": RANDOM_SEED,
    }
    return metrics, split_details


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson and Spearman correlations for each input/target pair."""
    rows: list[dict[str, float | str]] = []
    for input_name in INPUT_COLUMNS:
        for target_name in TARGET_COLUMNS:
            x = data[input_name].to_numpy(dtype=float)
            y = data[target_name].to_numpy(dtype=float)
            rows.append(
                {
                    "input": input_name,
                    "target": target_name,
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "spearman_r": float(
                        np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def one_factor_sensitivity() -> pd.DataFrame:
    """Vary each input over its full range while holding other inputs at midpoints."""
    ranges = {
        "TDP_W": (30.0, 250.0),
        "air_velocity_m_s": (0.5, 15.0),
        "k_tim_W_mK": (1.0, 12.0),
    }
    midpoint = {name: (lo + hi) / 2.0 for name, (lo, hi) in ranges.items()}
    rows: list[dict[str, float | str]] = []
    for varied_name, (low, high) in ranges.items():
        low_inputs = dict(midpoint)
        high_inputs = dict(midpoint)
        low_inputs[varied_name] = low
        high_inputs[varied_name] = high
        low_result = heat_sink_model(
            low_inputs["TDP_W"],
            low_inputs["air_velocity_m_s"],
            low_inputs["k_tim_W_mK"],
        )
        high_result = heat_sink_model(
            high_inputs["TDP_W"],
            high_inputs["air_velocity_m_s"],
            high_inputs["k_tim_W_mK"],
        )
        for target_name in TARGET_COLUMNS:
            delta = float(high_result[target_name] - low_result[target_name])
            rows.append(
                {
                    "target": target_name,
                    "varied_input": varied_name,
                    "low_value": low,
                    "high_value": high,
                    "signed_output_change": delta,
                    "absolute_output_change": abs(delta),
                }
            )
    sensitivity = pd.DataFrame(rows)
    sensitivity["rank_by_absolute_change"] = sensitivity.groupby("target")[
        "absolute_output_change"
    ].rank(method="min", ascending=False).astype(int)
    return sensitivity.sort_values(["target", "rank_by_absolute_change"])


def markdown_table(frame: pd.DataFrame, float_format: str = ".6g") -> str:
    """Render a small DataFrame as Markdown without adding a tabulate dependency."""
    headers = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return format(float(value), float_format)
        return str(value)

    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_report(
    output_dir: Path,
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    split_details: dict[str, Any],
) -> dict[str, Any]:
    """Save the reference-case results and a concise reproducible report."""
    reference = heat_sink_model(150.0, 1.0, 4.0)
    max_re = float(data["Re"].max())
    reference_result = {key: float(np.asarray(value)) for key, value in reference.items()}
    metrics_table = markdown_table(metrics)
    corr_table = markdown_table(correlations, float_format=".4f")
    sensitivity_table = markdown_table(sensitivity)

    results = {
        "dataset_rows": int(len(data)),
        "ambient_C": CONFIG.ambient_C,
        "input_ranges": {
            "TDP_W": [30.0, 250.0],
            "air_velocity_m_s": [0.5, 15.0],
            "k_tim_W_mK": [1.0, 12.0],
        },
        "config": asdict(CONFIG),
        "split": split_details,
        "maximum_reynolds_number": max_re,
        "turbulent_branch_used": bool(max_re >= CONFIG.reynolds_transition),
        "reference_case_150W_1m_s_4W_mK": reference_result,
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    corr_summary = []
    for target_name in TARGET_COLUMNS:
        subset = correlations.loc[correlations["target"] == target_name]
        strongest = subset.iloc[subset["spearman_r"].abs().argmax()]
        corr_summary.append(
            f"- For `{target_name}`, the strongest absolute Spearman correlation is "
            f"`{strongest['input']}` ({strongest['spearman_r']:.4f})."
        )

    report = f"""# Heat-sink surrogate assessment

## Assumptions

- Ambient temperature is {CONFIG.ambient_C:.1f} °C. Die, heat-sink geometry, aluminum conductivity, TIM thickness, junction-to-case resistance, and 25 °C air properties follow the supplied assessment values.
- Air properties are held constant at the supplied values (k = {CONFIG.air_conductivity_W_mK:g} W/m·K, ν = {CONFIG.air_kinematic_viscosity_m2_s:.3g} m²/s, Pr = {CONFIG.air_prandtl:g}); they are not recalculated at the 40 °C operating point.
- Total resistance is `R_jc + R_TIM + R_cond + R_conv`. The model includes a straight-fin efficiency approximation and corrected fin length to represent fin-tip convection.
- The source assessment's laminar developing-flow and turbulent Nusselt correlations are used with its fin-spacing Reynolds-number convention and Re transition of {CONFIG.reynolds_transition:.0f}.
- The systematic grid contains {len(data):,} combinations: 25 TDP values, 31 air velocities, and 25 TIM conductivities.

## Reference case and key results

At 150 W, 1 m/s, and k_TIM = 4 W/m·K, the model predicts R_total = {reference_result['R_total_C_per_W']:.6f} °C/W and T_j = {reference_result['T_junction_C']:.4f} °C. Maximum Re in the sweep is {max_re:.1f}; the turbulent branch is {'used' if max_re >= CONFIG.reynolds_transition else 'not reached'} over these requested ranges.

## Surrogate-model comparison

The test set holds out entire air-velocity levels. Scores measure agreement with this analytic model, not measured hardware accuracy.

{metrics_table}

## Correlation analysis

{corr_table}

{chr(10).join(corr_summary)}

## One-factor sensitivity

Each input is varied across its full requested range while the other inputs are at their midpoints. Positive values mean the output rises as the input moves from low to high.

{sensitivity_table}

## Limitations and improvements

The model assumes uniform forced airflow, constant material and air properties, and simplified fin efficiency. The Nusselt correlations and their range of validity should be reviewed by a thermal engineer. The surrogate is trained on synthetic labels from this model, so high ML scores do not prove real-device accuracy. Future work should compare against CFD and bench measurements, use temperature-dependent air properties and measured fan curves, test multiple heat-sink geometries, and use group holdouts across geometry and operating conditions.
"""
    (output_dir / "summary_report.md").write_text(report, encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            if "__file__" in globals()
            else Path.cwd()
        ),
        help="Directory for generated CSV and report files (default: script directory).",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = make_dataset()
    if len(data) < 500:
        raise RuntimeError("Generated dataset unexpectedly contains fewer than 500 rows.")
    if not np.isfinite(data.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Generated dataset contains non-finite numeric values.")
    data.to_csv(output_dir / "thermal_dataset.csv", index=False, float_format="%.10g")

    metrics, split_details = evaluate_models(data)
    correlations = correlation_analysis(data)
    sensitivity = one_factor_sensitivity()
    metrics.to_csv(output_dir / "model_metrics.csv", index=False, float_format="%.10g")
    correlations.to_csv(
        output_dir / "correlation_analysis.csv", index=False, float_format="%.10g"
    )
    sensitivity.to_csv(
        output_dir / "sensitivity_analysis.csv", index=False, float_format="%.10g"
    )
    results = write_report(
        output_dir, data, metrics, correlations, sensitivity, split_details
    )

    print(f"Generated {len(data):,} rows: {output_dir / 'thermal_dataset.csv'}")
    print("\nHeld-out model metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nOne-factor sensitivity (ranked within each target):")
    print(
        sensitivity[["target", "varied_input", "signed_output_change", "rank_by_absolute_change"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print(
        f"\nMax Reynolds number: {results['maximum_reynolds_number']:.2f}; "
        f"turbulent branch used: {results['turbulent_branch_used']}"
    )
    ref = results["reference_case_150W_1m_s_4W_mK"]
    print(
        f"Reference case: R_total={ref['R_total_C_per_W']:.6f} °C/W, "
        f"T_j={ref['T_junction_C']:.4f} °C"
    )


if __name__ == "__main__":
    main()
