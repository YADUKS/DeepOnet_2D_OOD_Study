"""
Preview one actual stored simple-training or OOD dataset case.

The preview never generates a geometry, solves a new FE problem, trains a
model, or performs prediction.  Every displayed field is loaded from the
selected stored dataset and its original FE case file.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.fem.geometry_spec import (
    OOD_GEOMETRY_FAMILIES,
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)
from src.analysis.style import MASK_CMAP, SDF_CMAP, TEMPERATURE_CMAP


def resolve_stored_preview_dataset_path(cfg):
    """Resolve and validate the configured stored family dataset."""
    group = str(cfg.preview_dataset_group).strip().lower()
    family = canonical_geometry_name(cfg.preview_dataset_family)
    size = int(cfg.preview_dataset_size)
    if size <= 0:
        raise ValueError("preview_dataset_size must be positive.")

    if group == "training":
        if family not in TRAINING_GEOMETRY_FAMILIES:
            raise ValueError(
                f"{family!r} is not a simple training geometry family."
            )
        path = cfg.geometry_dataset_path(family, size)
    elif group == "ood":
        if family not in OOD_GEOMETRY_FAMILIES:
            raise ValueError(f"{family!r} is not an OOD geometry family.")
        path = cfg.ood_dataset_path(family, size)
    else:
        raise ValueError("preview_dataset_group must be 'training' or 'ood'.")
    return group, family, size, path


def _scalar(data, name, cast):
    if name not in data:
        raise ValueError(f"Stored dataset is missing required field {name!r}.")
    values = np.asarray(data[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Stored field {name!r} must contain one value.")
    return cast(values[0])


def _resolve_fe_path(dataset_path, stored_fe_path):
    from src.data.design import resolve_reference
    return resolve_reference(dataset_path, stored_fe_path)


def load_stored_preview_case(cfg):
    """Load and validate the selected dataset case and original FE fields."""
    group, family, size, dataset_path = resolve_stored_preview_dataset_path(cfg)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Stored preview dataset does not exist: {dataset_path}"
        )
    case_number = int(cfg.preview_dataset_case_number)
    if case_number < 0:
        raise ValueError("preview_dataset_case_number cannot be negative.")

    with np.load(dataset_path, allow_pickle=True) as data:
        required = (
            "sdf",
            "mask",
            "temp_physical",
            "metadata_json",
            "fe_paths",
            "coords",
            "geometry_group",
            "geometry_family",
            "n_cases",
            "Nx",
            "Ny",
            "Lx",
            "Ly",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Stored dataset is missing fields: {missing}")
        stored_group = str(_scalar(data, "geometry_group", str))
        stored_family = canonical_geometry_name(
            _scalar(data, "geometry_family", str)
        )
        ncases = int(_scalar(data, "n_cases", int))
        Nx = int(_scalar(data, "Nx", int))
        Ny = int(_scalar(data, "Ny", int))
        Lx = float(_scalar(data, "Lx", float))
        Ly = float(_scalar(data, "Ly", float))
        if stored_group != group:
            raise ValueError(
                f"Requested group {group!r}, but dataset is {stored_group!r}."
            )
        if stored_family != family:
            raise ValueError(
                f"Requested family {family!r}, but dataset is {stored_family!r}."
            )
        if case_number >= ncases:
            raise IndexError(
                f"preview_dataset_case_number={case_number} is outside the "
                f"{ncases}-case {family} dataset."
            )
        sdf = np.asarray(data["sdf"][case_number], dtype=np.float32)
        mask = np.asarray(data["mask"][case_number], dtype=np.float32)
        temperature = np.asarray(
            data["temp_physical"][case_number],
            dtype=np.float32,
        )
        coords = np.asarray(data["coords"], dtype=np.float32)
        metadata_json = str(data["metadata_json"][case_number])
        fe_recorded_path = str(data["fe_paths"][case_number])
        dataset_role = str(data["roles"][case_number]) if "roles" in data else "legacy"
        case_id = (
            int(data["case_ids"][case_number])
            if "case_ids" in data
            else case_number
        )
        candidate_id = (
            int(data["candidate_ids"][case_number])
            if "candidate_ids" in data
            else -1
        )

    expected_points = Nx * Ny
    if sdf.shape != (expected_points,):
        raise ValueError(f"Stored SDF has invalid shape: {sdf.shape}.")
    if mask.shape != sdf.shape or temperature.shape != sdf.shape:
        raise ValueError("Stored mask/temperature shapes do not match SDF.")
    if coords.shape != (expected_points, 2):
        raise ValueError(f"Stored coordinate shape is invalid: {coords.shape}.")
    if not np.isfinite(sdf).all() or not np.isfinite(mask).all():
        raise ValueError("Stored SDF or mask contains non-finite values.")
    if not np.isfinite(temperature).all() or not np.isfinite(coords).all():
        raise ValueError("Stored temperature or coordinates are non-finite.")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("Stored material mask is not binary.")
    if np.count_nonzero((sdf >= 0.0).astype(np.float32) != mask):
        raise ValueError("Stored mask disagrees with the SDF sign.")

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("Stored geometry metadata is not valid JSON.") from error
    fe_path = _resolve_fe_path(dataset_path, fe_recorded_path)
    with np.load(fe_path, allow_pickle=True) as fe_data:
        required_fe = (
            "points",
            "triangles",
            "temperature",
            "boundary_points",
        )
        missing_fe = [name for name in required_fe if name not in fe_data]
        if missing_fe:
            raise ValueError(f"Stored FE case is missing fields: {missing_fe}")
        fe_points = np.asarray(fe_data["points"], dtype=np.float64)
        fe_triangles = np.asarray(fe_data["triangles"], dtype=np.int32)
        fe_temperature = np.asarray(
            fe_data["temperature"],
            dtype=np.float64,
        )
        boundary_points = np.asarray(
            fe_data["boundary_points"],
            dtype=np.float64,
        )
        area = (
            float(np.asarray(fe_data["area"]).reshape(-1)[0])
            if "area" in fe_data
            else np.nan
        )
        perimeter = (
            float(np.asarray(fe_data["perimeter"]).reshape(-1)[0])
            if "perimeter" in fe_data
            else np.nan
        )
        clearance = (
            float(np.asarray(fe_data["domain_clearance"]).reshape(-1)[0])
            if "domain_clearance" in fe_data
            else np.nan
        )

    if fe_points.ndim != 2 or fe_points.shape[1] != 2:
        raise ValueError("Stored FE points must have shape [N, 2].")
    if fe_triangles.ndim != 2 or fe_triangles.shape[1] != 3:
        raise ValueError("Stored FE triangles must have shape [M, 3].")
    if fe_temperature.shape != (len(fe_points),):
        raise ValueError("Stored FE temperature does not match FE vertices.")
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Stored boundary points must have shape [N, 2].")

    return {
        "group": group,
        "dataset_role": dataset_role,
        "family": family,
        "dataset_size": size,
        "case_number": case_number,
        "case_id": case_id,
        "candidate_id": candidate_id,
        "dataset_path": dataset_path,
        "fe_path": fe_path,
        "metadata": metadata,
        "sdf": sdf.reshape(Ny, Nx),
        "mask": mask.reshape(Ny, Nx),
        "temperature": temperature.reshape(Ny, Nx),
        "coords": coords,
        "Nx": Nx,
        "Ny": Ny,
        "Lx": Lx,
        "Ly": Ly,
        "fe_points": fe_points,
        "fe_triangles": fe_triangles,
        "fe_temperature": fe_temperature,
        "boundary_points": boundary_points,
        "area": area,
        "perimeter": perimeter,
        "domain_clearance": clearance,
    }


def _add_domain_and_boundary(axis, case):
    axis.plot(
        [0.0, case["Lx"], case["Lx"], 0.0, 0.0],
        [0.0, 0.0, case["Ly"], case["Ly"], 0.0],
        color="black",
        linewidth=1.5,
    )
    boundary = case["boundary_points"]
    closed = np.vstack((boundary, boundary[0]))
    axis.fill(boundary[:, 0], boundary[:, 1], color="white", zorder=2)
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color="tab:blue",
        linewidth=2.0,
        zorder=3,
    )
    axis.set_xlim(0.0, case["Lx"])
    axis.set_ylim(0.0, case["Ly"])
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def _finish_individual_axis(axis, case, title):
    axis.set_xlim(0.0, case["Lx"])
    axis.set_ylim(0.0, case["Ly"])
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)


def _save_individual_preview_figures(
    cfg,
    case,
    output_dir,
    extent,
    xgrid,
    ygrid,
    masked_temperature,
    triangulation,
):
    """Save each combined-preview panel as a standalone figure."""
    individual_dir = os.path.join(output_dir, "individual_figures")
    os.makedirs(individual_dir, exist_ok=True)
    saved_paths = []

    def save_figure(fig, stem):
        png_path = os.path.join(individual_dir, f"{stem}.png")
        fig.savefig(png_path, dpi=cfg.publish_dpi, bbox_inches="tight")
        saved_paths.append(png_path)
        if cfg.stored_preview_save_pdf:
            pdf_path = os.path.join(individual_dir, f"{stem}.pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            saved_paths.append(pdf_path)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    _add_domain_and_boundary(axis, case)
    axis.set_title("Original stored hollow geometry")
    save_figure(fig, "01_original_geometry")

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.triplot(triangulation, color="0.30", linewidth=0.35)
    _add_domain_and_boundary(axis, case)
    axis.set_title("Original NGSolve FE mesh")
    save_figure(fig, "02_fe_mesh")

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    image = axis.imshow(
        case["sdf"],
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=SDF_CMAP,
    )
    axis.contour(
        xgrid, ygrid, case["sdf"], levels=[0.0],
        colors="black", linewidths=1.2,
    )
    _finish_individual_axis(axis, case, "Stored signed-distance field")
    fig.colorbar(image, ax=axis, label="Normalized SDF")
    save_figure(fig, "03_sdf")

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    image = axis.imshow(
        case["mask"],
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=MASK_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    axis.contour(
        xgrid, ygrid, case["sdf"], levels=[0.0],
        colors="tab:blue", linewidths=1.2,
    )
    _finish_individual_axis(axis, case, "Stored material/hollow mask")
    fig.colorbar(image, ax=axis, label="0 = hollow, 1 = solid")
    save_figure(fig, "04_mask")

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    image = axis.imshow(
        masked_temperature,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=TEMPERATURE_CMAP,
    )
    axis.contour(
        xgrid, ygrid, case["sdf"], levels=[0.0],
        colors="cyan", linewidths=1.0,
    )
    _finish_individual_axis(axis, case, "Stored Cartesian temperature")
    fig.colorbar(image, ax=axis, label="Temperature [K]")
    save_figure(fig, "05_cartesian_temperature")

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    contour = axis.tricontourf(
        triangulation,
        case["fe_temperature"],
        levels=60,
        cmap=TEMPERATURE_CMAP,
    )
    _add_domain_and_boundary(axis, case)
    axis.set_title("Original FE temperature contour")
    fig.colorbar(contour, ax=axis, label="Temperature [K]")
    save_figure(fig, "06_fe_temperature")

    print(f"Saved individual preview figures: {individual_dir}")
    return saved_paths


def save_stored_geometry_preview(cfg, case):
    """Create the six-panel stored-case preview and metadata file."""
    output_dir = cfg.stored_preview_output_dir(
        case["group"],
        case["family"],
        case["dataset_size"],
        case["case_number"],
    )
    os.makedirs(output_dir, exist_ok=True)
    extent = [0.0, case["Lx"], 0.0, case["Ly"]]
    xgrid = np.linspace(0.0, case["Lx"], case["Nx"])
    ygrid = np.linspace(0.0, case["Ly"], case["Ny"])
    masked_temperature = case["temperature"].copy()
    masked_temperature[case["mask"] < 0.5] = np.nan

    triangulation = mtri.Triangulation(
        case["fe_points"][:, 0],
        case["fe_points"][:, 1],
        case["fe_triangles"],
    )
    triangle_mask = np.any(
        ~np.isfinite(case["fe_temperature"][case["fe_triangles"]]),
        axis=1,
    )
    triangulation.set_mask(triangle_mask)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    _add_domain_and_boundary(axes[0, 0], case)
    axes[0, 0].set_title("Original stored hollow geometry")

    axes[0, 1].triplot(
        triangulation,
        color="0.30",
        linewidth=0.35,
    )
    _add_domain_and_boundary(axes[0, 1], case)
    axes[0, 1].set_title("Original NGSolve FE mesh")

    sdf_image = axes[0, 2].imshow(
        case["sdf"],
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=SDF_CMAP,
    )
    axes[0, 2].contour(
        xgrid,
        ygrid,
        case["sdf"],
        levels=[0.0],
        colors="black",
        linewidths=1.2,
    )
    axes[0, 2].set_title("Stored signed-distance field")
    axes[0, 2].set_xlabel("x [m]")
    axes[0, 2].set_ylabel("y [m]")
    fig.colorbar(sdf_image, ax=axes[0, 2], label="Normalized SDF")

    mask_image = axes[1, 0].imshow(
        case["mask"],
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=MASK_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].contour(
        xgrid,
        ygrid,
        case["sdf"],
        levels=[0.0],
        colors="tab:blue",
        linewidths=1.2,
    )
    axes[1, 0].set_title("Stored material/hollow mask")
    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("y [m]")
    fig.colorbar(mask_image, ax=axes[1, 0], label="0 = hollow, 1 = solid")

    temp_image = axes[1, 1].imshow(
        masked_temperature,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=TEMPERATURE_CMAP,
    )
    axes[1, 1].contour(
        xgrid,
        ygrid,
        case["sdf"],
        levels=[0.0],
        colors="cyan",
        linewidths=1.0,
    )
    axes[1, 1].set_title("Stored Cartesian temperature")
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("y [m]")
    fig.colorbar(temp_image, ax=axes[1, 1], label="Temperature [K]")

    fe_contour = axes[1, 2].tricontourf(
        triangulation,
        case["fe_temperature"],
        levels=60,
        cmap=TEMPERATURE_CMAP,
    )
    _add_domain_and_boundary(axes[1, 2], case)
    axes[1, 2].set_title("Original FE temperature contour")
    fig.colorbar(fe_contour, ax=axes[1, 2], label="Temperature [K]")

    fig.suptitle(
        f"Stored {case['group']} geometry: {case['family']} | "
        f"dataset case {case['case_number']} | case ID {case['case_id']}",
        fontsize=15,
    )
    png_path = os.path.join(output_dir, "preview_combined.png")
    pdf_path = os.path.join(output_dir, "preview_combined.pdf")
    fig.savefig(png_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    if cfg.stored_preview_save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    if cfg.stored_preview_show:
        plt.show()
    else:
        plt.close(fig)
    save_individual = bool(cfg.stored_preview_save_individual)
    individual_paths = []
    if save_individual:
        individual_paths = _save_individual_preview_figures(
            cfg,
            case,
            output_dir,
            extent,
            xgrid,
            ygrid,
            masked_temperature,
            triangulation,
        )

    metadata_payload = {
        "group": case["group"],
        "dataset_role": case["dataset_role"],
        "family": case["family"],
        "dataset_size": case["dataset_size"],
        "dataset_case_number": case["case_number"],
        "stored_case_id": case["case_id"],
        "stored_candidate_id": case["candidate_id"],
        "dataset_path": case["dataset_path"],
        "fe_path": case["fe_path"],
        "geometry_metadata": case["metadata"],
        "area": case["area"],
        "perimeter": case["perimeter"],
        "domain_clearance": case["domain_clearance"],
        "grid": {"Nx": case["Nx"], "Ny": case["Ny"]},
        "domain": {"Lx": case["Lx"], "Ly": case["Ly"]},
        "cartesian_temperature_min": float(np.nanmin(masked_temperature)),
        "cartesian_temperature_max": float(np.nanmax(masked_temperature)),
        "fe_temperature_min": float(np.nanmin(case["fe_temperature"])),
        "fe_temperature_max": float(np.nanmax(case["fe_temperature"])),
        "individual_figures_saved": bool(save_individual),
        "individual_figure_paths": individual_paths,
    }
    metadata_path = os.path.join(output_dir, "preview_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata_payload, stream, indent=2)
    print(f"Saved stored geometry preview: {png_path}")
    print(f"Saved stored geometry metadata: {metadata_path}")
    return {
        "output_dir": output_dir,
        "png_path": png_path,
        "pdf_path": pdf_path if cfg.stored_preview_save_pdf else None,
        "metadata_path": metadata_path,
        "individual_paths": individual_paths,
    }


def preview_stored_geometry(cfg):
    """Load and render the configured stored training or OOD case."""
    case = load_stored_preview_case(cfg)
    #save_individual = bool(cfg.stored_preview_save_individual)
    return save_stored_geometry_preview(
        cfg,
        case
        #save_individual=save_individual,
    )
