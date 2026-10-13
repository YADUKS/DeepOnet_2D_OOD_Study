"""Export an exact stored training, composed, or OOD FE case for ANSYS."""

import csv
import json
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.data.compose import get_composition_id
from src.fem.geometry_spec import (
    OOD_GEOMETRY_FAMILIES,
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)


def _scalar(data, name):
    return np.asarray(data[name]).reshape(-1)[0]


def _resolve_fe_path(dataset_path, stored_fe_path):
    from src.data.design import resolve_reference
    return resolve_reference(dataset_path, stored_fe_path)


def _composed_id(cfg):
    return (
        str(cfg.ansys_export_composition_id).strip()
        or str(cfg.training_composition_id).strip()
        or get_composition_id(cfg)
    )


def resolve_ansys_export_dataset_path(cfg):
    """Resolve the configured reusable dataset without generating new data."""
    group = str(cfg.ansys_export_dataset_group).strip().lower()
    family = canonical_geometry_name(cfg.ansys_export_family)
    size = int(cfg.ansys_export_dataset_size)
    if group == "training":
        if family not in TRAINING_GEOMETRY_FAMILIES:
            raise ValueError(f"{family!r} is not a training geometry family.")
        if size <= 0:
            raise ValueError("ansys_export_dataset_size must be positive.")
        return group, family, cfg.geometry_dataset_path(family, size)
    if group == "ood":
        if family not in OOD_GEOMETRY_FAMILIES:
            raise ValueError(f"{family!r} is not an OOD geometry family.")
        if size <= 0:
            raise ValueError("ansys_export_dataset_size must be positive.")
        return group, family, cfg.ood_dataset_path(family, size)
    if group == "composed":
        if family not in TRAINING_GEOMETRY_FAMILIES:
            raise ValueError(
                f"{family!r} is not a composed simple-geometry family."
            )
        composition_id = _composed_id(cfg)
        return group, family, cfg.composed_dataset_path(composition_id)
    raise ValueError(
        "ansys_export_dataset_group must be 'training', 'composed', or 'ood'."
    )


def load_ansys_export_case(cfg):
    """Load one exact stored FE case and its generation metadata."""
    group, requested_family, dataset_path = resolve_ansys_export_dataset_path(cfg)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ANSYS export dataset is missing: {dataset_path}")
    case_number = int(cfg.ansys_export_case_number)
    if case_number < 0:
        raise ValueError("ansys_export_case_number cannot be negative.")

    with np.load(dataset_path, allow_pickle=True) as data:
        required = ("fe_paths", "metadata_json")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Dataset is missing ANSYS export fields: {missing}")
        n_cases = len(data["fe_paths"])
        dataset_case_number = case_number
        if group == "composed":
            if "geometry_family" not in data:
                raise ValueError(
                    "Composed dataset is missing the geometry_family field."
                )
            geometry_families = np.asarray(
                [
                    canonical_geometry_name(value)
                    for value in data["geometry_family"]
                ],
                dtype=object,
            )
            family_indices = np.flatnonzero(
                geometry_families == requested_family
            )
            if len(family_indices) == 0:
                raise ValueError(
                    f"Composed dataset contains no {requested_family!r} cases."
                )
            if case_number >= len(family_indices):
                raise IndexError(
                    f"Composed {requested_family} case {case_number} is "
                    f"outside [0, {len(family_indices) - 1}]."
                )
            dataset_case_number = int(family_indices[case_number])
        elif case_number >= n_cases:
            raise IndexError(
                f"ANSYS export case {case_number} is outside [0, {n_cases - 1}]."
            )

        metadata = json.loads(
            str(data["metadata_json"][dataset_case_number])
        )
        family = requested_family
        fe_path = _resolve_fe_path(
            dataset_path,
            data["fe_paths"][dataset_case_number],
        )
        case_id_field = "composed_case_id" if group == "composed" else "case_ids"
        case_id = (
            int(data[case_id_field][dataset_case_number])
            if case_id_field in data
            else dataset_case_number
        )
        source_case_id = (
            int(data["source_case_id"][dataset_case_number])
            if "source_case_id" in data
            else case_id
        )
        source_dataset_path = (
            str(data["source_dataset_path"][dataset_case_number])
            if "source_dataset_path" in data
            else dataset_path
        )
        physics = {
            name: float(_scalar(data, name))
            for name in ("Lx", "Ly", "kappa", "h_hole", "T_inf", "q_top")
            if name in data
        }

    with np.load(fe_path, allow_pickle=True) as fe_data:
        required = ("points", "triangles", "boundary_points")
        missing = [name for name in required if name not in fe_data]
        if missing:
            raise ValueError(f"FE case is missing ANSYS export fields: {missing}")
        points = np.asarray(fe_data["points"], dtype=np.float64)
        triangles = np.asarray(fe_data["triangles"], dtype=np.int64)
        hollow_boundary = np.asarray(
            fe_data["boundary_points"],
            dtype=np.float64,
        )

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("FE points must have shape [n_points, >=2].")
    points = points[:, :2]
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("FE triangles must have shape [n_triangles, 3].")
    if triangles.size == 0 or triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("FE triangle connectivity is invalid.")
    if hollow_boundary.ndim != 2 or hollow_boundary.shape[1] != 2:
        raise ValueError("Stored hollow boundary must have shape [n_points, 2].")
    if not np.isfinite(points).all() or not np.isfinite(hollow_boundary).all():
        raise ValueError("Stored FE geometry contains non-finite coordinates.")

    physics.setdefault("Lx", float(cfg.Lx))
    physics.setdefault("Ly", float(cfg.Ly))
    physics.setdefault("kappa", float(cfg.kappa))
    physics.setdefault("h_hole", float(cfg.h_hole))
    physics.setdefault("T_inf", float(cfg.T_inf))
    physics.setdefault("q_top", float(cfg.q_top))
    return {
        "group": group,
        "family": family,
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_size": n_cases,
        "case_number": case_number,
        "composed_dataset_case_number": (
            dataset_case_number if group == "composed" else None
        ),
        "case_id": case_id,
        "source_case_id": source_case_id,
        "source_dataset_path": source_dataset_path,
        "fe_path": fe_path,
        "metadata": metadata,
        "points": points,
        "triangles": triangles,
        "hollow_boundary": hollow_boundary,
        "physics": physics,
        "composition_id": _composed_id(cfg) if group == "composed" else None,
    }


def _oriented_triangles(points, triangles):
    triangles = triangles.copy()
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    signed_twice_area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    if np.any(np.isclose(signed_twice_area, 0.0, atol=1e-18)):
        raise ValueError("FE mesh contains degenerate triangles.")
    reverse = signed_twice_area < 0.0
    triangles[reverse, 1], triangles[reverse, 2] = (
        triangles[reverse, 2].copy(),
        triangles[reverse, 1].copy(),
    )
    return triangles


def _boundary_edges(triangles):
    """Return oriented material-boundary edges from CCW planar triangles."""
    oriented = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    undirected = np.sort(oriented, axis=1)
    _, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return oriented[counts[inverse] == 1]


def build_extruded_surface(case, thickness_m):
    """Extrude the exact 2D FE mesh into a closed triangular surface."""
    thickness_m = float(thickness_m)
    if not np.isfinite(thickness_m) or thickness_m <= 0.0:
        raise ValueError("ansys_extrusion_thickness must be positive.")
    points_2d = case["points"]
    planar = _oriented_triangles(points_2d, case["triangles"])
    n_points = len(points_2d)
    vertices = np.vstack(
        (
            np.column_stack((points_2d, np.zeros(n_points))),
            np.column_stack(
                (points_2d, np.full(n_points, thickness_m))
            ),
        )
    )
    bottom = planar[:, [0, 2, 1]]
    top = planar + n_points
    boundary = _boundary_edges(planar)
    a = boundary[:, 0]
    b = boundary[:, 1]
    sides = np.vstack(
        (
            np.column_stack((a, b, b + n_points)),
            np.column_stack((a, b + n_points, a + n_points)),
        )
    )
    facets = np.vstack((bottom, top, sides)).astype(np.int64)
    return vertices, facets, boundary


def _validate_closed_surface(facets):
    all_edges = np.vstack(
        (
            facets[:, [0, 1]],
            facets[:, [1, 2]],
            facets[:, [2, 0]],
        )
    )
    _, counts = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Extruded STL surface is not watertight.")


def _write_ascii_stl(path, vertices, facets, solid_name):
    with open(path, "w", encoding="ascii", newline="\n") as stream:
        stream.write(f"solid {solid_name}\n")
        for facet in facets:
            xyz = vertices[facet]
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            norm = np.linalg.norm(normal)
            if norm <= 0.0:
                raise ValueError("STL contains a zero-area facet.")
            normal /= norm
            stream.write(
                "  facet normal "
                f"{normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n"
            )
            stream.write("    outer loop\n")
            for point in xyz:
                stream.write(
                    "      vertex "
                    f"{point[0]:.9e} {point[1]:.9e} {point[2]:.9e}\n"
                )
            stream.write("    endloop\n  endfacet\n")
        stream.write(f"endsolid {solid_name}\n")


def _classify_boundary_edges(points, edges, Lx, Ly):
    tolerance = max(Lx, Ly) * 1e-6
    records = []
    for edge_id, (start_id, end_id) in enumerate(edges):
        start = points[start_id]
        end = points[end_id]
        if np.all(np.abs([start[1] - Ly, end[1] - Ly]) <= tolerance):
            boundary = "top_heat_flux"
        elif np.all(np.abs([start[1], end[1]]) <= tolerance):
            boundary = "bottom_adiabatic"
        elif np.all(np.abs([start[0], end[0]]) <= tolerance):
            boundary = "left_adiabatic"
        elif np.all(np.abs([start[0] - Lx, end[0] - Lx]) <= tolerance):
            boundary = "right_adiabatic"
        else:
            boundary = "hollow_convection"
        records.append((edge_id, boundary, start_id, end_id, *start, *end))
    return records


def _write_boundary_csv(path, records, scale):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "edge_id",
                "boundary_name",
                "start_vertex_id",
                "end_vertex_id",
                "x1",
                "y1",
                "x2",
                "y2",
            )
        )
        for record in records:
            writer.writerow((*record[:4], *(scale * np.asarray(record[4:]))))


def export_stored_geometry_for_ansys(cfg):
    """Export the configured stored case as STL plus ANSYS support files."""
    case = load_ansys_export_case(cfg)
    thickness_m = float(cfg.ansys_extrusion_thickness)
    vertices_m, facets, boundary_edges = build_extruded_surface(
        case,
        thickness_m,
    )
    _validate_closed_surface(facets)

    units = str(cfg.ansys_export_units).strip().lower()
    unit_scales = {"m": 1.0, "mm": 1000.0}
    if units not in unit_scales:
        raise ValueError("ansys_export_units must be 'm' or 'mm'.")
    scale = unit_scales[units]
    vertices = vertices_m * scale
    output_dir = cfg.ansys_export_output_dir(
        case["group"],
        case["family"],
        case["dataset_size"],
        case["case_number"],
        case["composition_id"],
    )
    os.makedirs(output_dir, exist_ok=True)

    stl_path = os.path.join(output_dir, "plate_with_hollow_cutout.stl")
    _write_ascii_stl(
        stl_path,
        vertices,
        facets,
        f"{case['group']}_{case['family']}_case_{case['case_number']:06d}",
    )

    records = _classify_boundary_edges(
        case["points"],
        boundary_edges,
        case["physics"]["Lx"],
        case["physics"]["Ly"],
    )
    boundary_csv_path = os.path.join(output_dir, "boundary_edges.csv")
    _write_boundary_csv(boundary_csv_path, records, scale)
    hollow_csv_path = os.path.join(output_dir, "hollow_boundary.csv")
    with open(hollow_csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("point_id", f"x_{units}", f"y_{units}"))
        for point_id, point in enumerate(case["hollow_boundary"] * scale):
            writer.writerow((point_id, point[0], point[1]))

    metadata = {
        "export_schema_version": 1,
        "dataset_group": case["group"],
        "geometry_family": case["family"],
        "dataset_size": case["dataset_size"],
        "dataset_case_number": case["case_number"],
        "composed_dataset_case_number": case["composed_dataset_case_number"],
        "stored_case_id": case["case_id"],
        "source_case_id": case["source_case_id"],
        "composition_id": case["composition_id"],
        "dataset_path": case["dataset_path"],
        "source_dataset_path": case["source_dataset_path"],
        "fe_path": case["fe_path"],
        "geometry_record": case["metadata"],
        "coordinate_units": units,
        "stl_extrusion_thickness": thickness_m * scale,
        "stl_is_watertight": True,
        "stl_vertices": int(len(vertices)),
        "stl_triangular_facets": int(len(facets)),
        "physics_si_units": case["physics"],
        "ansys_boundary_mapping": {
            "top_extruded_side": "heat flux q_top",
            "hollow_extruded_side": "convection h_hole, T_inf",
            "left_right_bottom_extruded_sides": "adiabatic",
            "front_and_back_faces": "adiabatic for 2D equivalence",
        },
    }
    metadata_path = os.path.join(output_dir, "geometry_parameters.json")
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    conditions_path = os.path.join(output_dir, "ansys_boundary_conditions.txt")
    with open(conditions_path, "w", encoding="utf-8") as stream:
        stream.write("ANSYS steady-state thermal boundary conditions\n")
        stream.write("All values below use SI units.\n\n")
        stream.write(
            f"Thermal conductivity: {case['physics']['kappa']:.12g} W/(m K)\n"
        )
        stream.write(
            f"Top extruded face heat flux: {case['physics']['q_top']:.12g} W/m^2\n"
        )
        stream.write(
            "Hollow extruded face convection coefficient: "
            f"{case['physics']['h_hole']:.12g} W/(m^2 K)\n"
        )
        stream.write(
            f"Hollow convection ambient temperature: "
            f"{case['physics']['T_inf']:.12g} K\n"
        )
        stream.write("Left, right, and bottom extruded faces: adiabatic\n")
        stream.write("Front and back faces: adiabatic for 2D equivalence\n")

    triangulation = mtri.Triangulation(
        case["points"][:, 0] * scale,
        case["points"][:, 1] * scale,
        case["triangles"],
    )
    fig, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    axis.triplot(triangulation, color="0.45", linewidth=0.3)
    closed = np.vstack((case["hollow_boundary"], case["hollow_boundary"][0]))
    axis.plot(closed[:, 0] * scale, closed[:, 1] * scale, color="tab:blue", lw=2)
    axis.set_aspect("equal")
    axis.set_xlabel(f"x [{units}]")
    axis.set_ylabel(f"y [{units}]")
    axis.set_title(
        f"ANSYS export | {case['group']} {case['family']} | "
        f"case {case['case_number']}"
    )
    preview_path = os.path.join(output_dir, "geometry_preview.png")
    fig.savefig(preview_path, dpi=cfg.publish_dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved watertight ANSYS STL: {stl_path}")
    print(f"Saved ANSYS export metadata: {metadata_path}")
    return {
        "output_dir": output_dir,
        "stl_path": stl_path,
        "boundary_csv_path": boundary_csv_path,
        "hollow_csv_path": hollow_csv_path,
        "metadata_path": metadata_path,
        "conditions_path": conditions_path,
        "preview_path": preview_path,
    }
