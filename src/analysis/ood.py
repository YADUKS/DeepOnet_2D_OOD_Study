"""
Evaluate a trained simple-geometry DeepONet on separate OOD families.

The evaluator always uses the temperature scaler fitted on the training split.
OOD-local T_min/T_max values are deliberately ignored.
"""

import csv
import json
import os
import platform
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from src.data.design import resolve_reference
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.fem.geometry_spec import OOD_GEOMETRY_FAMILIES, canonical_geometry_name
from src.models.deeponet import SimpleDeepONet
from src.analysis.style import ERROR_CMAP, TEMPERATURE_CMAP
from src.data.protocol import apply_temperature_normalization


OOD_METRIC_FIELDS = (
    "family",
    "case_id",
    "candidate_id",
    "masked_mse_normalized",
    "mae_kelvin",
    "rmse_kelvin",
    "relative_l2",
    "maximum_absolute_error_kelvin",
    "mean_relative_error_percent",
    "fe_mse_kelvin_squared",
    "fe_mae_kelvin",
    "fe_rmse_kelvin",
    "fe_relative_l2",
    "fe_maximum_absolute_error_kelvin",
    "fe_mean_relative_error_percent",
)


def _finite_color_limits(values, vmin=None, vmax=None):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot create a colorbar for an all-NaN field.")
    lower = float(finite.min()) if vmin is None else float(vmin)
    upper = float(finite.max()) if vmax is None else float(vmax)
    if not upper > lower:
        padding = max(abs(lower) * 1e-9, 1e-12)
        lower -= padding
        upper += padding
    return lower, upper


def _full_height_colorbar(
    fig,
    axis,
    mappable,
    label,
    vmin,
    vmax,
    extend="neither",
):
    """Add an axis-height colorbar whose ticks include both endpoints."""
    divider = make_axes_locatable(axis)
    color_axis = divider.append_axes("right", size="5%", pad=0.08)
    colorbar = fig.colorbar(mappable, cax=color_axis, extend=extend)
    colorbar.set_label(label)
    colorbar.set_ticks(np.linspace(float(vmin), float(vmax), 7))
    return colorbar


def _hide_field_axis_frame(axis):
    """Show only the contour field; retain its adjacent colorbar."""
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


from src.training.runtime import get_device


def load_training_temperature_scaler(cfg):
    """Load and validate the scaler fitted on training solid points."""
    path = cfg.temperature_normalization_path
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training temperature scaler does not exist: {path}"
        )
    with np.load(path, allow_pickle=True) as data:
        required = ("method", "fitted_on", "temperature_min", "temperature_max")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Temperature scaler is missing fields: {missing}")
        method = str(np.asarray(data["method"]).reshape(-1)[0])
        fitted_on = str(np.asarray(data["fitted_on"]).reshape(-1)[0])
        temperature_min = float(
            np.asarray(data["temperature_min"]).reshape(-1)[0]
        )
        temperature_max = float(
            np.asarray(data["temperature_max"]).reshape(-1)[0]
        )
        training_dataset_path = (
            str(np.asarray(data["training_dataset_path"]).reshape(-1)[0])
            if "training_dataset_path" in data
            else "unknown"
        )
        training_dataset_sha256 = str(data["training_dataset_sha256"][0]) if "training_dataset_sha256" in data else None
    if method != "training_solid_minmax":
        raise ValueError(
            "OOD evaluation requires a composed-model scaler fitted by "
            f"training_solid_minmax; found {method!r}."
        )
    if fitted_on != "training_split_solid_points":
        raise ValueError(
            "OOD evaluation requires a scaler fitted only on training solid "
            f"points; found {fitted_on!r}."
        )
    if not np.isfinite([temperature_min, temperature_max]).all():
        raise ValueError("Training temperature scaler contains non-finite values.")
    if temperature_max <= temperature_min:
        raise ValueError("Training temperature scaler has a non-positive range.")
    return {
        "path": path,
        "method": method,
        "fitted_on": fitted_on,
        "temperature_min": temperature_min,
        "temperature_max": temperature_max,
        "training_dataset_path": resolve_reference(path, training_dataset_path),
        "training_dataset_sha256": training_dataset_sha256,
    }


def load_training_model_settings(cfg):
    """Load architecture/grid settings recorded by the completed training run."""
    path = cfg.config_used_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training configuration snapshot is missing: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    settings = payload.get("settings", {})
    required = ("latent_dim", "Nx", "Ny", "Lx", "Ly")
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError(f"Training configuration is missing settings: {missing}")
    return {
        "path": path,
        "latent_dim": int(settings["latent_dim"]),
        "Nx": int(settings["Nx"]),
        "Ny": int(settings["Ny"]),
        "Lx": float(settings["Lx"]),
        "Ly": float(settings["Ly"]),
        "settings": settings,
    }


def _scalar(data, name, cast):
    if name not in data:
        raise ValueError(f"OOD dataset is missing required field {name!r}.")
    values = np.asarray(data[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"OOD field {name!r} must contain one value.")
    return cast(values[0])


def load_and_validate_ood_dataset(cfg, family, training_settings):
    """Load one OOD family and verify it matches the trained model grid."""
    family = canonical_geometry_name(family)
    if family not in OOD_GEOMETRY_FAMILIES:
        raise ValueError(f"{family!r} is not an OOD family.")
    if family not in cfg.ood_evaluation_source_sizes:
        raise KeyError(f"No OOD evaluation source size configured for {family!r}.")
    source_size = int(cfg.ood_evaluation_source_sizes[family])
    path = cfg.ood_dataset_path(family, source_size)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OOD dataset does not exist: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = (
            "sdf",
            "mask",
            "temp_physical",
            "coords",
            "geometry_group",
            "geometry_family",
            "training_eligible",
            "fe_paths",
            "n_cases",
            "Nx",
            "Ny",
            "Lx",
            "Ly",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"OOD dataset {path} is missing fields: {missing}")
        stored_group = str(_scalar(data, "geometry_group", str))
        stored_family = canonical_geometry_name(
            _scalar(data, "geometry_family", str)
        )
        training_eligible = bool(_scalar(data, "training_eligible", bool))
        if stored_group != "ood" or training_eligible:
            raise ValueError(f"Dataset {path} is not safely marked as OOD-only.")
        if stored_family != family:
            raise ValueError(
                f"Expected OOD family {family!r}, found {stored_family!r}."
            )
        for name in ("kappa", "h_hole", "T_inf", "q_top"):
            if name not in data or not np.isclose(_scalar(data, name, float), training_settings["settings"][name]):
                raise ValueError(f"OOD {name} differs from the trained thermal problem.")

        sdf = np.asarray(data["sdf"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.float32)
        temp_physical = np.asarray(data["temp_physical"], dtype=np.float32)
        coords = np.asarray(data["coords"], dtype=np.float32)
        ncases = int(_scalar(data, "n_cases", int))
        Nx = int(_scalar(data, "Nx", int))
        Ny = int(_scalar(data, "Ny", int))
        Lx = float(_scalar(data, "Lx", float))
        Ly = float(_scalar(data, "Ly", float))
        case_ids = (
            np.asarray(data["case_ids"], dtype=np.int32)
            if "case_ids" in data
            else np.arange(ncases, dtype=np.int32)
        )
        candidate_ids = (
            np.asarray(data["candidate_ids"], dtype=np.int32)
            if "candidate_ids" in data
            else np.full(ncases, -1, dtype=np.int32)
        )
        metadata_json = (
            np.asarray(data["metadata_json"], dtype=object)
            if "metadata_json" in data
            else np.full(ncases, "{}", dtype=object)
        )
        fe_paths = np.asarray([resolve_reference(path, p) for p in data["fe_paths"]], dtype=object)

    expected_points = Nx * Ny
    if ncases <= 0:
        raise ValueError(f"OOD dataset {path} contains no cases.")
    if sdf.shape != (ncases, expected_points):
        raise ValueError(f"Invalid OOD SDF shape in {path}: {sdf.shape}.")
    if mask.shape != sdf.shape or temp_physical.shape != sdf.shape:
        raise ValueError(f"OOD mask/temperature shapes do not match SDF in {path}.")
    if coords.shape != (expected_points, 2):
        raise ValueError(f"Invalid OOD coordinate shape in {path}: {coords.shape}.")
    if not np.isfinite(sdf).all() or not np.isfinite(mask).all():
        raise ValueError(f"OOD SDF or mask contains non-finite values in {path}.")
    if not np.isfinite(temp_physical).all() or not np.isfinite(coords).all():
        raise ValueError(
            f"OOD temperature or coordinates contain non-finite values in {path}."
        )
    if fe_paths.shape != (ncases,):
        raise ValueError(f"OOD fe_paths must contain one path per case in {path}.")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError(f"OOD mask is not binary in {path}.")
    if np.count_nonzero((sdf >= 0.0).astype(np.float32) != mask):
        raise ValueError(f"OOD mask disagrees with the SDF sign in {path}.")
    if (
        Nx != training_settings["Nx"]
        or Ny != training_settings["Ny"]
        or not np.isclose(Lx, training_settings["Lx"])
        or not np.isclose(Ly, training_settings["Ly"])
    ):
        raise ValueError(
            f"OOD grid/domain in {path} does not match the trained model."
        )

    coords_normalized = coords.copy()
    coords_normalized[:, 0] /= Lx
    coords_normalized[:, 1] /= Ly
    return {
        "path": path,
        "family": family,
        "sdf": sdf,
        "mask": mask,
        "temp_physical": temp_physical,
        "coords_normalized": coords_normalized.astype(np.float32),
        "case_ids": case_ids,
        "candidate_ids": candidate_ids,
        "metadata_json": metadata_json,
        "fe_paths": fe_paths,
        "ncases": ncases,
        "Nx": Nx,
        "Ny": Ny,
        "Lx": Lx,
        "Ly": Ly,
    }


def select_ood_test_cases(dataset, requested_case_numbers):
    """Select exact zero-based stored case numbers from one OOD family."""
    family = dataset["family"]
    case_numbers = tuple(int(value) for value in requested_case_numbers)
    if not case_numbers:
        raise ValueError(f"Explicit OOD test selection for {family!r} is empty.")
    if len(case_numbers) != len(set(case_numbers)):
        raise ValueError(f"OOD test selection for {family!r} contains duplicates.")
    invalid = [
        value
        for value in case_numbers
        if value < 0 or value >= dataset["ncases"]
    ]
    if invalid:
        raise IndexError(
            f"OOD {family} case numbers {invalid} are outside "
            f"[0, {dataset['ncases'] - 1}]."
        )
    indices = np.asarray(case_numbers, dtype=np.int64)
    records = [
        {
            "ood_dataset_case_number": int(case_number),
            "stored_case_id": int(dataset["case_ids"][case_number]),
            "candidate_id": int(dataset["candidate_ids"][case_number]),
        }
        for case_number in case_numbers
    ]
    for key in (
        "sdf",
        "mask",
        "temp_physical",
        "case_ids",
        "candidate_ids",
        "metadata_json",
        "fe_paths",
    ):
        dataset[key] = dataset[key][indices]
    dataset["ncases"] = len(indices)
    dataset["selected_original_case_numbers"] = indices
    return dataset, records


def build_case_dependent_trunk_input(coords_t, sdf_batch):
    if coords_t.dim() != 2 or coords_t.shape[1] != 2:
        raise ValueError("OOD coordinates must contain normalized [x, y].")
    if sdf_batch.dim() != 2 or sdf_batch.shape[1] != coords_t.shape[0]:
        raise ValueError("OOD SDF batch does not match the coordinate grid.")
    coords_batch = coords_t.unsqueeze(0).expand(sdf_batch.shape[0], -1, -1)
    return torch.cat((coords_batch, sdf_batch.unsqueeze(-1)), dim=-1)


def predict_ood_dataset(model, dataset, device, batch_size):
    """Predict normalized temperatures in deterministic case order."""
    coords_t = torch.tensor(
        dataset["coords_normalized"],
        dtype=torch.float32,
        device=device,
    )
    predictions = []
    batch_size = max(int(batch_size), 1)
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["ncases"], batch_size):
            stop = min(start + batch_size, dataset["ncases"])
            sdf_batch = torch.tensor(
                dataset["sdf"][start:stop],
                dtype=torch.float32,
                device=device,
            )
            trunk_batch = build_case_dependent_trunk_input(
                coords_t,
                sdf_batch,
            )
            predictions.append(model(sdf_batch, trunk_batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def compute_case_metrics(
    family,
    case_id,
    candidate_id,
    prediction_normalized,
    truth_normalized,
    prediction_physical,
    truth_physical,
    mask,
    fe_metrics=None,
):
    valid = mask > 0.5
    if not np.any(valid):
        raise ValueError(f"OOD {family} case {case_id} contains no solid points.")
    normalized_error = prediction_normalized[valid] - truth_normalized[valid]
    physical_error = prediction_physical[valid] - truth_physical[valid]
    truth_valid = truth_physical[valid]
    absolute_error = np.abs(physical_error)
    relative_point_error = 100.0 * absolute_error / (
        np.abs(truth_valid) + 1e-12
    )
    row = {
        "family": family,
        "case_id": int(case_id),
        "candidate_id": int(candidate_id),
        "masked_mse_normalized": float(np.mean(normalized_error ** 2)),
        "mae_kelvin": float(np.mean(absolute_error)),
        "rmse_kelvin": float(np.sqrt(np.mean(physical_error ** 2))),
        "relative_l2": float(
            np.linalg.norm(physical_error)
            / (np.linalg.norm(truth_valid) + 1e-12)
        ),
        "maximum_absolute_error_kelvin": float(absolute_error.max()),
        "mean_relative_error_percent": float(relative_point_error.mean()),
    }
    if fe_metrics is None:
        fe_metrics = {
            "fe_mse_kelvin_squared": np.nan,
            "fe_mae_kelvin": np.nan,
            "fe_rmse_kelvin": np.nan,
            "fe_relative_l2": np.nan,
            "fe_maximum_absolute_error_kelvin": np.nan,
            "fe_mean_relative_error_percent": np.nan,
        }
    row.update(fe_metrics)
    return row


def bilinear_interpolate_regular_grid(field, Lx, Ly, xq, yq):
    """Interpolate a ``[Ny, Nx]`` Cartesian field at FE mesh vertices."""
    field = np.asarray(field, dtype=np.float64)
    xgrid = np.linspace(0.0, float(Lx), field.shape[1])
    ygrid = np.linspace(0.0, float(Ly), field.shape[0])
    xq = np.clip(np.asarray(xq, dtype=np.float64), xgrid[0], xgrid[-1])
    yq = np.clip(np.asarray(yq, dtype=np.float64), ygrid[0], ygrid[-1])
    ix = np.clip(
        np.searchsorted(xgrid, xq, side="right") - 1,
        0,
        len(xgrid) - 2,
    )
    iy = np.clip(
        np.searchsorted(ygrid, yq, side="right") - 1,
        0,
        len(ygrid) - 2,
    )
    x0, x1 = xgrid[ix], xgrid[ix + 1]
    y0, y1 = ygrid[iy], ygrid[iy + 1]
    tx = (xq - x0) / (x1 - x0 + 1e-15)
    ty = (yq - y0) / (y1 - y0 + 1e-15)
    return (
        (1.0 - tx) * (1.0 - ty) * field[iy, ix]
        + tx * (1.0 - ty) * field[iy, ix + 1]
        + (1.0 - tx) * ty * field[iy + 1, ix]
        + tx * ty * field[iy + 1, ix + 1]
    )


def evaluate_prediction_on_fe_mesh(fe_path, prediction_grid, Lx, Ly):
    """Load original FE truth and interpolate a Cartesian prediction to it."""
    resolved_path = os.fspath(fe_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Stored FE case does not exist: {resolved_path}")
    with np.load(resolved_path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
        truth_fe = np.asarray(data["temperature"], dtype=np.float64)
        boundary_points = (
            np.asarray(data["boundary_points"], dtype=np.float64)
            if "boundary_points" in data
            else None
        )
    prediction_fe = bilinear_interpolate_regular_grid(
        prediction_grid,
        Lx,
        Ly,
        points[:, 0],
        points[:, 1],
    )
    valid = np.isfinite(truth_fe) & np.isfinite(prediction_fe)
    if not np.any(valid):
        raise ValueError(f"FE case contains no comparable values: {resolved_path}")
    error = prediction_fe[valid] - truth_fe[valid]
    absolute = np.abs(error)
    truth_valid = truth_fe[valid]
    relative = 100.0 * absolute / (np.abs(truth_valid) + 1e-12)
    metrics = {
        "fe_mse_kelvin_squared": float(np.mean(error ** 2)),
        "fe_mae_kelvin": float(np.mean(absolute)),
        "fe_rmse_kelvin": float(np.sqrt(np.mean(error ** 2))),
        "fe_relative_l2": float(
            np.linalg.norm(error) / (np.linalg.norm(truth_valid) + 1e-12)
        ),
        "fe_maximum_absolute_error_kelvin": float(absolute.max()),
        "fe_mean_relative_error_percent": float(relative.mean()),
    }
    return {
        "path": resolved_path,
        "points": points,
        "triangles": triangles,
        "boundary_points": boundary_points,
        "truth": truth_fe,
        "prediction": prediction_fe,
        "metrics": metrics,
    }


def save_metrics_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OOD_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def get_selected_case_output_dir(
    family_dir,
    test_case_number,
    case_id,
):
    """Create the same per-selected-case folder convention as legacy."""
    case_dir = os.path.join(
        family_dir,
        f"test_case_number_{int(test_case_number):04d}_"
        f"case_id_{int(case_id):06d}",
    )
    os.makedirs(case_dir, exist_ok=True)
    return case_dir


def _plot_exact_stored_geometry(axis, boundary_points, Lx, Ly):
    """Draw the physical domain and exact stored hollow without an outline."""
    axis.fill(
        [0.0, Lx, Lx, 0.0],
        [0.0, 0.0, Ly, Ly],
        color="0.88",
        edgecolor="none",
    )
    if boundary_points is None:
        raise ValueError(
            "Exact geometry plotting requires boundary_points in the FE file."
        )
    boundary_points = np.asarray(boundary_points, dtype=np.float64)
    if (
        boundary_points.ndim != 2
        or boundary_points.shape[1] != 2
        or len(boundary_points) < 3
        or not np.isfinite(boundary_points).all()
    ):
        raise ValueError("Stored boundary_points are invalid.")
    axis.fill(
        boundary_points[:, 0],
        boundary_points[:, 1],
        color="white",
        edgecolor="none",
    )
    axis.plot(
        [0.0, Lx, Lx, 0.0, 0.0],
        [0.0, 0.0, Ly, Ly, 0.0],
        color="black",
        linewidth=1.5,
    )
    axis.set_xlim(0.0, Lx)
    axis.set_ylim(0.0, Ly)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def save_legacy_style_case_bundle(
    cfg,
    dataset,
    case_number,
    test_case_number,
    truth_physical,
    prediction_physical,
    fe_result,
    metrics,
    output_dir,
    evaluation_label,
    save_pdf,
):
    """Save the legacy 3x3 figure and per-case data bundle."""
    family = dataset["family"]
    case_id = int(dataset["case_ids"][case_number])
    Ny, Nx = dataset["Ny"], dataset["Nx"]
    Lx, Ly = dataset["Lx"], dataset["Ly"]
    sdf = dataset["sdf"][case_number].reshape(Ny, Nx)
    mask = dataset["mask"][case_number].reshape(Ny, Nx)
    truth_grid = truth_physical[case_number].reshape(Ny, Nx).copy()
    prediction_grid = prediction_physical[case_number].reshape(Ny, Nx).copy()
    truth_grid[mask < 0.5] = np.nan
    prediction_grid[mask < 0.5] = np.nan
    xgrid = np.linspace(0.0, Lx, Nx)
    ygrid = np.linspace(0.0, Ly, Ny)

    points = fe_result["points"]
    triangles = fe_result["triangles"]
    boundary_points = fe_result["boundary_points"]
    true_fe = fe_result["truth"]
    pred_fe = fe_result["prediction"]
    valid = np.isfinite(true_fe) & np.isfinite(pred_fe)
    signed_fe = np.full_like(true_fe, np.nan, dtype=np.float64)
    signed_fe[valid] = pred_fe[valid] - true_fe[valid]
    absolute_fe = np.abs(signed_fe)
    relative_fe = 100.0 * absolute_fe / (np.abs(true_fe) + 1e-12)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    tri.set_mask(np.any(~valid[triangles], axis=1))

    temperature_min = float(np.nanmin(true_fe))
    temperature_max = float(np.nanmax(true_fe))
    signed_limit = max(float(np.nanmax(np.abs(signed_fe))), 1e-12)
    absolute_min = float(np.nanmin(absolute_fe))
    absolute_max = float(np.nanmax(absolute_fe))
    relative_min = float(np.nanmin(relative_fe))
    relative_max = float(np.nanmax(relative_fe))

    # AxesDivider colorbars and constrained_layout compete for the same space.
    # Reserve explicit gutters so labels cannot overlap the adjacent panel.
    fig, axes = plt.subplots(3, 3, figsize=(18, 11.5))
    fig.subplots_adjust(left=0.05, right=0.94, bottom=0.06, top=0.90,
                        wspace=0.48, hspace=0.45)
    fig.suptitle(f"{evaluation_label}: {family} | held-out case {test_case_number}", fontsize=15)
    _plot_exact_stored_geometry(
        axes[0, 0], boundary_points, Lx, Ly
    )
    axes[0, 0].set_title("Stored geometry", fontsize=11)

    axes[0, 1].triplot(tri, lw=0.5, color="tab:blue")
    axes[0, 1].set_xlim(0.0, Lx)
    axes[0, 1].set_ylim(0.0, Ly)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].set_title("Original FE mesh", fontsize=11)

    panels = (
        (
            axes[0, 2], "fe", true_fe, "Ground truth on FE mesh",
            "Temperature [K]", TEMPERATURE_CMAP, temperature_min,
            temperature_max, "neither",
        ),
        (
            axes[1, 0], "fe", pred_fe, "Predicted on FE mesh",
            "Temperature [K]", TEMPERATURE_CMAP,
            temperature_min, temperature_max, "neither",
        ),
        (
            axes[1, 1], "grid", truth_grid, "Ground truth on grid",
            "Temperature [K]", TEMPERATURE_CMAP, temperature_min,
            temperature_max, "neither",
        ),
        (
            axes[1, 2], "grid", prediction_grid, "Predicted on grid",
            "Temperature [K]", TEMPERATURE_CMAP,
            temperature_min, temperature_max, "neither",
        ),
        (
            axes[2, 0], "fe", signed_fe, "Signed error on FE mesh",
            "Signed error [K]", ERROR_CMAP, -signed_limit, signed_limit,
            "neither",
        ),
        (
            axes[2, 1], "fe", absolute_fe, "Absolute error on FE mesh",
            "Absolute error [K]", ERROR_CMAP, absolute_min, absolute_max,
            "neither",
        ),
        (
            axes[2, 2], "fe", relative_fe, "Relative error on FE mesh",
            "Relative error [%]", ERROR_CMAP, relative_min, relative_max,
            "neither",
        ),
    )
    for (
        axis,
        grid_type,
        values,
        title,
        label,
        cmap,
        vmin,
        vmax,
        extend,
    ) in panels:
        if grid_type == "fe":
            image = axis.tripcolor(
                tri, values, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax,
            )
        else:
            image = axis.contourf(
                xgrid, ygrid, values, levels=np.linspace(vmin, vmax, 61), cmap=cmap,
                vmin=vmin, vmax=vmax, extend="both",
            )
        axis.set_xlim(0.0, Lx)
        axis.set_ylim(0.0, Ly)
        axis.set_aspect("equal")
        _hide_field_axis_frame(axis)
        axis.set_title(title, fontsize=11, pad=10)
        colorbar = _full_height_colorbar(
            fig, axis, image, label, vmin, vmax, extend=extend
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(label, fontsize=9)

    prefix = (
        f"test_case_number_{int(test_case_number):04d}_"
        f"case_id_{case_id:06d}"
    )
    combined_path = os.path.join(output_dir, f"{prefix}_combined.png")
    fig.savefig(combined_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(
            os.path.join(output_dir, f"{prefix}_combined.pdf"),
            bbox_inches="tight",
        )
    plt.close(fig)

    # Standalone domain and hollow-geometry figure.
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    _plot_exact_stored_geometry(axis, boundary_points, Lx, Ly)
    geometry_path = os.path.join(
        output_dir,
        f"{prefix}_geometry_with_domain.png",
    )
    fig.savefig(geometry_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(
            os.path.join(
                output_dir,
                f"{prefix}_geometry_with_domain.pdf",
            ),
            bbox_inches="tight",
        )
    plt.close(fig)

    # Standalone original unstructured FE-mesh figure.
    mesh_triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.triplot(mesh_triangulation, color="0.35", linewidth=0.4)
    axis.plot(
        [0.0, Lx, Lx, 0.0, 0.0],
        [0.0, 0.0, Ly, Ly, 0.0],
        color="black",
        linewidth=1.2,
    )
    axis.set_xlim(0.0, Lx)
    axis.set_ylim(0.0, Ly)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    mesh_path = os.path.join(output_dir, f"{prefix}_fe_mesh.png")
    fig.savefig(mesh_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(
            os.path.join(output_dir, f"{prefix}_fe_mesh.pdf"),
            bbox_inches="tight",
        )
    plt.close(fig)

    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    save_metrics_csv(metrics_path, [metrics])
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    metadata = json.loads(str(dataset["metadata_json"][case_number]))
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "evaluation_type": evaluation_label,
                "family_test_case_number": int(test_case_number),
                "case_id": case_id,
                "candidate_id": int(dataset["candidate_ids"][case_number]),
                "geometry_metadata": metadata,
                "fe_source": fe_result["path"],
                "metrics": metrics,
            },
            stream,
            indent=2,
        )
    return {
        "combined_path": combined_path,
        "geometry_path": geometry_path,
        "mesh_path": mesh_path,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
    }


def summarize_metrics(rows):
    numeric_fields = OOD_METRIC_FIELDS[3:]
    summary = {"number_of_cases": len(rows)}
    for field in numeric_fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        summary[f"mean_{field}"] = float(values.mean())
        summary[f"median_{field}"] = float(np.median(values))
        summary[f"maximum_{field}"] = float(values.max())
    return summary


def save_family_comparison_figure(
    cfg,
    rows,
    family_order,
    output_root=None,
    filename_prefix="ood",
    save_pdf=None,
):
    """Save boxplots comparing errors across the selected OOD families."""
    relative_l2 = [
        [row["relative_l2"] for row in rows if row["family"] == family]
        for family in family_order
    ]
    mae_kelvin = [
        [row["mae_kelvin"] for row in rows if row["family"] == family]
        for family in family_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].boxplot(relative_l2, labels=family_order, showmeans=True)
    axes[0].set_ylabel("Relative L2")
    axes[1].boxplot(mae_kelvin, labels=family_order, showmeans=True)
    axes[1].set_ylabel("MAE [K]")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=25)
    output_root = output_root or cfg.ood_evaluation_root
    if save_pdf is None:
        save_pdf = cfg.ood_save_pdf_figures
    path = os.path.join(
        output_root,
        f"{filename_prefix}_family_error_comparison.png",
    )
    fig.savefig(path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(
            os.path.join(
                output_root,
                f"{filename_prefix}_family_error_comparison.pdf",
            ),
            bbox_inches="tight",
        )
    plt.close(fig)


def save_rmse_mape_plot_and_csv(
    cfg,
    rows,
    output_root,
    filename_prefix,
    save_pdf=False,
):
    """Save legacy-style RMSE [K] and MAPE [%] values for all evaluated cases."""
    if not rows:
        return

    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(
        output_root, f"{filename_prefix}_rmse_mape_values.csv"
    )
    png_path = os.path.join(
        output_root, f"{filename_prefix}_rmse_mape_plot.png"
    )
    pdf_path = os.path.join(
        output_root, f"{filename_prefix}_rmse_mape_plot.pdf"
    )

    evaluated_case_numbers = np.arange(1, len(rows) + 1, dtype=np.int64)
    rmse_values = np.asarray(
        [row["rmse_kelvin"] for row in rows], dtype=np.float64
    )
    mape_values = np.asarray(
        [row["mean_relative_error_percent"] for row in rows],
        dtype=np.float64,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "evaluated_case_number",
                "family",
                "case_id",
                "rmse_K",
                "mape_percent",
            )
        )
        for evaluated_number, row, rmse, mape in zip(
            evaluated_case_numbers, rows, rmse_values, mape_values
        ):
            writer.writerow(
                (
                    int(evaluated_number),
                    row["family"],
                    int(row["case_id"]),
                    float(rmse),
                    float(mape),
                )
            )

    worst_rmse = int(np.nanargmax(rmse_values))
    worst_mape = int(np.nanargmax(mape_values))
    fig, rmse_axis = plt.subplots(figsize=(11, 6))
    rmse_axis.plot(
        evaluated_case_numbers,
        rmse_values,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="RMSE [K]",
    )
    rmse_axis.set_xlabel("Evaluated test case number")
    rmse_axis.set_ylabel("RMSE [K]")
    rmse_axis.grid(True, alpha=0.3)

    mape_axis = rmse_axis.twinx()
    mape_axis.plot(
        evaluated_case_numbers,
        mape_values,
        marker="s",
        markersize=3,
        linewidth=1.5,
        linestyle="--",
        label="MAPE [%]",
    )
    mape_axis.set_ylabel("MAPE [%]")
    rmse_axis.scatter(
        evaluated_case_numbers[worst_rmse],
        rmse_values[worst_rmse],
        s=80,
        color="red",
        zorder=5,
    )
    mape_axis.scatter(
        evaluated_case_numbers[worst_mape],
        mape_values[worst_mape],
        s=80,
        color="black",
        zorder=5,
    )
    rmse_lines, rmse_labels = rmse_axis.get_legend_handles_labels()
    mape_lines, mape_labels = mape_axis.get_legend_handles_labels()
    rmse_axis.legend(
        rmse_lines + mape_lines,
        rmse_labels + mape_labels,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_case_figure(
    cfg,
    dataset,
    case_number,
    truth_physical,
    prediction_physical,
    output_dir=None,
    evaluation_label="OOD",
    save_pdf=None,
):
    """Save ground truth, prediction, and error maps on the Cartesian grid."""
    family = dataset["family"]
    case_id = int(dataset["case_ids"][case_number])
    Ny, Nx = dataset["Ny"], dataset["Nx"]
    mask = dataset["mask"][case_number].reshape(Ny, Nx)
    sdf = dataset["sdf"][case_number].reshape(Ny, Nx)
    truth = truth_physical[case_number].reshape(Ny, Nx).copy()
    prediction = prediction_physical[case_number].reshape(Ny, Nx).copy()
    truth[mask < 0.5] = np.nan
    prediction[mask < 0.5] = np.nan
    signed = prediction - truth
    absolute = np.abs(signed)
    relative = 100.0 * absolute / (np.abs(truth) + 1e-12)
    extent = [0.0, dataset["Lx"], 0.0, dataset["Ly"]]
    common_min = float(np.nanmin(truth))
    common_max = float(np.nanmax(truth))
    signed_limit = max(float(np.nanmax(np.abs(signed))), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = (
        (
            truth, "FE ground truth", "Temperature [K]",
            TEMPERATURE_CMAP, common_min, common_max, "neither",
        ),
        (
            prediction, "DeepONet prediction", "Temperature [K]",
            TEMPERATURE_CMAP,
            common_min, common_max, "neither",
        ),
        (
            signed,
            "Signed error",
            "Signed error [K]",
            ERROR_CMAP,
            -signed_limit,
            signed_limit,
            "neither",
        ),
        (
            absolute, "Absolute error", "Absolute error [K]",
            ERROR_CMAP, None, None, "neither",
        ),
        (
            relative, "Relative error", "Relative error [%]",
            ERROR_CMAP, None, None, "neither",
        ),
    )
    for axis, panel in zip(axes.ravel(), panels):
        values, title, colorbar_label, cmap, vmin, vmax, extend = panel
        vmin, vmax = _finite_color_limits(values, vmin, vmax)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        _hide_field_axis_frame(axis)
        _full_height_colorbar(
            fig, axis, image, colorbar_label, vmin, vmax, extend=extend
        )

    axes.ravel()[-1].axis("off")
    family_dir = output_dir or cfg.ood_evaluation_family_dir(family)
    if save_pdf is None:
        save_pdf = cfg.ood_save_pdf_figures
    os.makedirs(family_dir, exist_ok=True)
    stem = os.path.join(family_dir, f"case_{case_id:06d}_comparison")
    fig.savefig(f"{stem}.png", dpi=cfg.publish_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_single_fe_figure(
    points,
    triangles,
    values,
    title,
    colorbar_label,
    path,
    cmap,
    vmin=None,
    vmax=None,
    save_pdf=True,
    dpi=300,
    extend="neither",
):
    valid_values = np.asarray(values, dtype=np.float64)
    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )
    triangle_mask = np.any(~np.isfinite(valid_values[triangles]), axis=1)
    triangulation.set_mask(triangle_mask)
    vmin, vmax = _finite_color_limits(valid_values, vmin, vmax)
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    contour = axis.tripcolor(
        triangulation,
        valid_values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_aspect("equal")
    _hide_field_axis_frame(axis)
    _full_height_colorbar(
        fig, axis, contour, colorbar_label, vmin, vmax, extend=extend
    )
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_original_style_fe_outputs(
    cfg,
    output_dir,
    evaluation_label,
    family,
    case_id,
    fe_result,
    metrics,
    save_pdf,
    geometry_metadata=None,
    test_case_number=None,
    source_case_id=None,
    domain=None,
):
    """Save original-style FE truth, prediction, and error deliverables."""
    os.makedirs(output_dir, exist_ok=True)
    points = fe_result["points"]
    triangles = fe_result["triangles"]
    truth = fe_result["truth"]
    prediction = fe_result["prediction"]
    valid = np.isfinite(truth) & np.isfinite(prediction)
    signed = np.full_like(truth, np.nan, dtype=np.float64)
    absolute = np.full_like(truth, np.nan, dtype=np.float64)
    relative = np.full_like(truth, np.nan, dtype=np.float64)
    signed[valid] = prediction[valid] - truth[valid]
    absolute[valid] = np.abs(signed[valid])
    relative[valid] = (
        100.0 * absolute[valid] / (np.abs(truth[valid]) + 1e-12)
    )
    common_min = float(np.nanmin(truth))
    common_max = float(np.nanmax(truth))
    signed_limit = max(float(np.nanmax(np.abs(signed))), 1e-12)
    stem = os.path.join(output_dir, f"case_{int(case_id):06d}")

    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )
    triangle_mask = np.any(
        ~(np.isfinite(truth[triangles]) & np.isfinite(prediction[triangles])),
        axis=1,
    )
    triangulation.set_mask(triangle_mask)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = (
        (
            truth, "FE ground truth", "Temperature [K]",
            TEMPERATURE_CMAP, common_min, common_max, "neither",
        ),
        (
            prediction,
            "DeepONet prediction on FE mesh",
            "Temperature [K]",
            TEMPERATURE_CMAP,
            common_min,
            common_max,
            "neither",
        ),
        (
            signed,
            "Signed FE error",
            "Signed error [K]",
            ERROR_CMAP,
            -signed_limit,
            signed_limit,
            "neither",
        ),
        (
            absolute, "Absolute FE error", "Absolute error [K]",
            ERROR_CMAP, None, None, "neither",
        ),
        (
            relative, "Relative FE error", "Relative error [%]",
            ERROR_CMAP, None, None, "neither",
        ),
    )
    for axis, (
        values,
        title,
        colorbar_label,
        cmap,
        vmin,
        vmax,
        extend,
    ) in zip(
        axes.ravel(),
        panels,
    ):
        vmin, vmax = _finite_color_limits(values, vmin, vmax)
        contour = axis.tripcolor(
            triangulation,
            values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_aspect("equal")
        _hide_field_axis_frame(axis)
        _full_height_colorbar(
            fig, axis, contour, colorbar_label, vmin, vmax, extend=extend
        )
    axes.ravel()[-1].axis("off")
    fig.savefig(
        f"{stem}_fe_comparison.png",
        dpi=cfg.publish_dpi,
        bbox_inches="tight",
    )
    if save_pdf:
        fig.savefig(f"{stem}_fe_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    separate_panels = (
        (
            truth,
            "FE ground truth",
            "Temperature [K]",
            "ground_truth_fe",
            TEMPERATURE_CMAP,
            common_min,
            common_max,
            "neither",
        ),
        (
            prediction,
            "DeepONet prediction on FE mesh",
            "Temperature [K]",
            "predicted_fe",
            TEMPERATURE_CMAP,
            common_min,
            common_max,
            "neither",
        ),
        (
            signed,
            "Signed FE error",
            "Error [K]",
            "signed_error_fe",
            ERROR_CMAP,
            -signed_limit,
            signed_limit,
            "neither",
        ),
        (
            absolute,
            "Absolute FE error",
            "Absolute error [K]",
            "absolute_error_fe",
            ERROR_CMAP,
            None,
            None,
            "neither",
        ),
        (
            relative,
            "Relative FE error",
            "Relative error [%]",
            "relative_error_fe",
            ERROR_CMAP,
            None,
            None,
            "neither",
        ),
    )
    for (
        values,
        title,
        label,
        suffix,
        cmap,
        vmin,
        vmax,
        extend,
    ) in separate_panels:
        _save_single_fe_figure(
            points,
            triangles,
            values,
            f"{evaluation_label}: {family} case {int(case_id)} - {title}",
            label,
            f"{stem}_{suffix}",
            cmap,
            vmin=vmin,
            vmax=vmax,
            save_pdf=save_pdf,
            dpi=cfg.publish_dpi,
            extend=extend,
        )

    summary_path = f"{stem}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as stream:
        stream.write(f"Evaluation type: {evaluation_label}\n")
        stream.write(f"Geometry family: {family}\n")
        if test_case_number is not None:
            stream.write(
                f"Selected test case number: {int(test_case_number)}\n"
            )
        stream.write(f"Case ID: {int(case_id)}\n")
        if source_case_id is not None:
            stream.write(f"Source family case ID: {int(source_case_id)}\n")
        stream.write(f"FE source: {fe_result['path']}\n")
        if domain is not None:
            stream.write(f"Domain Lx [m]: {float(domain['Lx']):.12e}\n")
            stream.write(f"Domain Ly [m]: {float(domain['Ly']):.12e}\n")
        if geometry_metadata is not None:
            stream.write("\nGeometry reconstruction record\n")
            stream.write("----------------------------------------\n")
            stream.write(
                json.dumps(
                    geometry_metadata,
                    indent=2,
                    sort_keys=True,
                )
            )
            stream.write("\n")
        boundary_points = fe_result.get("boundary_points")
        if boundary_points is not None:
            stream.write("\nExact stored hollow boundary points [m]\n")
            stream.write("----------------------------------------\n")
            stream.write(
                json.dumps(
                    np.asarray(
                        boundary_points,
                        dtype=np.float64,
                    ).tolist(),
                    indent=2,
                )
            )
            stream.write("\n")
        stream.write("\nPrediction error metrics\n")
        stream.write("----------------------------------------\n")
        for name in OOD_METRIC_FIELDS[3:]:
            stream.write(f"{name}: {float(metrics[name]):.12e}\n")
    return summary_path


def evaluate_ood_families(cfg):
    """Evaluate all selected OOD families and save metrics/figures/provenance."""
    if str(cfg.training_dataset_mode).strip().lower() != "composed":
        raise ValueError("OOD evaluation requires training_dataset_mode='composed'.")
    if not os.path.isfile(cfg.model_path):
        raise FileNotFoundError(f"Trained model does not exist: {cfg.model_path}")

    os.makedirs(cfg.ood_evaluation_root, exist_ok=True)
    device = get_device(cfg)
    scaler = load_training_temperature_scaler(cfg)
    training_settings = load_training_model_settings(cfg)

    selected_families = tuple(
        canonical_geometry_name(value)
        for value in cfg.ood_evaluation_families
    )
    if not selected_families:
        raise ValueError("No OOD evaluation families are selected.")
    invalid = [
        family for family in selected_families
        if family not in OOD_GEOMETRY_FAMILIES
    ]
    if invalid:
        raise ValueError(f"Unsupported OOD evaluation families: {invalid}")

    datasets = [
        load_and_validate_ood_dataset(cfg, family, training_settings)
        for family in selected_families
    ]
    requested_case_numbers = {
        canonical_geometry_name(family): tuple(int(value) for value in values)
        for family, values in dict(cfg.ood_test_case_numbers).items()
    }
    resolved_case_selection = {}
    selected_datasets = []
    for dataset in datasets:
        family = dataset["family"]
        if family in requested_case_numbers:
            dataset, records = select_ood_test_cases(
                dataset,
                requested_case_numbers[family],
            )
            resolved_case_selection[family] = records
        else:
            resolved_case_selection[family] = [
                {
                    "ood_dataset_case_number": int(case_number),
                    "stored_case_id": int(dataset["case_ids"][case_number]),
                    "candidate_id": int(dataset["candidate_ids"][case_number]),
                }
                for case_number in range(dataset["ncases"])
            ]
        selected_datasets.append(dataset)
    datasets = selected_datasets
    branch_features = training_settings["Nx"] * training_settings["Ny"]
    model = SimpleDeepONet(
        branch_in=branch_features,
        trunk_in=3,
        latent=training_settings["latent_dim"],
    ).to(device)
    try:
        state_dict = torch.load(
            cfg.model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_rows = []
    family_summaries = {}
    evaluated_sources = {}
    temperature_min = scaler["temperature_min"]
    temperature_max = scaler["temperature_max"]
    temperature_range = temperature_max - temperature_min

    for dataset in datasets:
        family = dataset["family"]
        truth_normalized = apply_temperature_normalization(
            dataset["temp_physical"],
            dataset["mask"],
            temperature_min,
            temperature_max,
        )
        prediction_normalized = predict_ood_dataset(
            model,
            dataset,
            device,
            cfg.ood_evaluation_batch_size,
        )
        # Match the legacy evaluator: keep this field unmasked for FE
        # interpolation. Display and metric routines apply the material mask.
        prediction_physical = (
            prediction_normalized * temperature_range + temperature_min
        )

        family_dir = cfg.ood_evaluation_family_dir(family)
        os.makedirs(family_dir, exist_ok=True)
        test_number_by_case_id = {
            int(record["stored_case_id"]): int(
                record["ood_dataset_case_number"]
            )
            for record in resolved_case_selection[family]
        }
        if family in requested_case_numbers:
            selected_plot_numbers = set(range(dataset["ncases"]))
        else:
            # A missing explicit selection means batch evaluation: calculate
            # every case, but do not create any per-case figure bundles.
            selected_plot_numbers = set()
        rows = []
        for case_number in range(dataset["ncases"]):
            prediction_grid = prediction_physical[case_number].reshape(
                dataset["Ny"],
                dataset["Nx"],
            )
            fe_result = evaluate_prediction_on_fe_mesh(
                dataset["fe_paths"][case_number],
                prediction_grid,
                dataset["Lx"],
                dataset["Ly"],
            )
            row = compute_case_metrics(
                family=family,
                case_id=dataset["case_ids"][case_number],
                candidate_id=dataset["candidate_ids"][case_number],
                prediction_normalized=prediction_normalized[case_number],
                truth_normalized=truth_normalized[case_number],
                prediction_physical=prediction_physical[case_number],
                truth_physical=dataset["temp_physical"][case_number],
                mask=dataset["mask"][case_number],
                fe_metrics=fe_result["metrics"],
            )
            rows.append(row)
            if case_number in selected_plot_numbers:
                case_id = int(dataset["case_ids"][case_number])
                test_case_number = test_number_by_case_id[case_id]
                case_dir = get_selected_case_output_dir(
                    family_dir,
                    test_case_number,
                    case_id,
                )
                save_case_figure(
                    cfg,
                    dataset,
                    case_number,
                    dataset["temp_physical"],
                    prediction_physical,
                    output_dir=case_dir,
                )
                save_original_style_fe_outputs(
                    cfg=cfg,
                    output_dir=case_dir,
                    evaluation_label="OOD test",
                    family=family,
                    case_id=case_id,
                    fe_result=fe_result,
                    metrics=row,
                    save_pdf=cfg.ood_save_pdf_figures,
                    geometry_metadata=json.loads(
                        str(dataset["metadata_json"][case_number])
                    ),
                    test_case_number=test_case_number,
                    source_case_id=case_id,
                    domain={
                        "Lx": dataset["Lx"],
                        "Ly": dataset["Ly"],
                    },
                )
                save_legacy_style_case_bundle(
                    cfg=cfg,
                    dataset=dataset,
                    case_number=case_number,
                    test_case_number=test_case_number,
                    truth_physical=dataset["temp_physical"],
                    prediction_physical=prediction_physical,
                    fe_result=fe_result,
                    metrics=row,
                    output_dir=case_dir,
                    evaluation_label="OOD test",
                    save_pdf=cfg.ood_save_pdf_figures,
                )

        save_metrics_csv(os.path.join(family_dir, "metrics.csv"), rows)
        save_rmse_mape_plot_and_csv(
            cfg,
            rows,
            output_root=family_dir,
            filename_prefix=f"all_{family}_test_cases",
            save_pdf=cfg.ood_save_pdf_figures,
        )
        family_summary = summarize_metrics(rows)
        with open(
            os.path.join(family_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(family_summary, stream, indent=2)

        all_rows.extend(rows)
        family_summaries[family] = family_summary
        evaluated_sources[family] = dataset["path"]
        print(
            f"OOD {family}: {dataset['ncases']} cases, "
            f"mean relative L2={family_summary['mean_relative_l2']:.6e}"
        )

    combined_metrics_path = os.path.join(
        cfg.ood_evaluation_root,
        "all_ood_metrics.csv",
    )
    save_metrics_csv(combined_metrics_path, all_rows)
    save_family_comparison_figure(cfg, all_rows, selected_families)
    overall_summary = summarize_metrics(all_rows)
    summary_payload = {
        "overall": overall_summary,
        "families": family_summaries,
    }
    with open(
        os.path.join(cfg.ood_evaluation_root, "ood_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary_payload, stream, indent=2)

    provenance = {
        "model_path": cfg.model_path,
        "training_config_path": training_settings["path"],
        "temperature_scaler": scaler,
        "ood_sources": evaluated_sources,
        "selected_families": list(selected_families),
        "requested_ood_test_case_numbers": {
            family: list(values)
            for family, values in requested_case_numbers.items()
        },
        "resolved_ood_test_case_selection": resolved_case_selection,
        "branch_input": "complete flattened SDF field",
        "trunk_input": ["x_normalized", "y_normalized", "sdf(x,y)"],
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "evaluation_config": asdict(cfg),
    }
    with open(
        os.path.join(cfg.ood_evaluation_root, "evaluation_provenance.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(provenance, stream, indent=2)

    print(f"Saved combined OOD metrics: {combined_metrics_path}")
    return summary_payload
