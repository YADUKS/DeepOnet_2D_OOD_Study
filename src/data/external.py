"""
Generate separate deterministic datasets for OOD hollow geometry families.

OOD files are never read by the simple-family training composer.
"""


from src.fem.geometry_spec import (
    GeometryCase,
    OOD_GEOMETRY_FAMILIES,
    canonical_geometry_name,
    transform_points,
)
from src.fem.ood_geometries import build_canonical_ood_boundary
from src.fem.sampling import sample_ood_geometry_case


OOD_DATASET_SCHEMA_VERSION = 2


from src.data.design import parameter_candidates, ROLE_SEED_OFFSETS


def _linear(unit_value, lower, upper):
    return float(lower) + float(unit_value) * (float(upper) - float(lower))


def _parameter_count(family):
    return {
        "star": 5,
        "aircraft": 8,
        "leaf": 7,
        "fin_like": 8,
    }[family]


def _sample_parameters(cfg, family, values, candidate_id):
    if family == "star":
        choices = tuple(int(value) for value in cfg.star_points)
        parameters = {
            "points": choices[candidate_id % len(choices)],
            "outer_radius": _linear(
                values[0],
                cfg.star_outer_radius_min,
                cfg.star_outer_radius_max,
            ),
            "inner_ratio": _linear(
                values[1],
                cfg.star_inner_ratio_min,
                cfg.star_inner_ratio_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = values[2:5]
    elif family == "aircraft":
        parameters = {
            "length": _linear(
                values[0],
                cfg.aircraft_length_min,
                cfg.aircraft_length_max,
            ),
            "wingspan": _linear(
                values[1],
                cfg.aircraft_wingspan_min,
                cfg.aircraft_wingspan_max,
            ),
            "fuselage_width_ratio": _linear(
                values[2],
                cfg.aircraft_fuselage_width_ratio_min,
                cfg.aircraft_fuselage_width_ratio_max,
            ),
            "tailspan_ratio": _linear(
                values[3],
                cfg.aircraft_tailspan_ratio_min,
                cfg.aircraft_tailspan_ratio_max,
            ),
            "wing_sweep": _linear(
                values[4],
                cfg.aircraft_wing_sweep_min,
                cfg.aircraft_wing_sweep_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = values[5:8]
    elif family == "leaf":
        parameters = {
            "length": _linear(
                values[0],
                cfg.leaf_length_min,
                cfg.leaf_length_max,
            ),
            "width": _linear(
                values[1],
                cfg.leaf_width_min,
                cfg.leaf_width_max,
            ),
            "curvature": _linear(
                values[2],
                cfg.leaf_curvature_min,
                cfg.leaf_curvature_max,
            ),
            "sharpness": _linear(
                values[3],
                cfg.leaf_sharpness_min,
                cfg.leaf_sharpness_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = values[4:7]
    elif family == "fin_like":
        choices = tuple(int(value) for value in cfg.fin_counts)
        parameters = {
            "height": _linear(
                values[0],
                cfg.fin_height_min,
                cfg.fin_height_max,
            ),
            "spine_width": _linear(
                values[1],
                cfg.fin_spine_width_min,
                cfg.fin_spine_width_max,
            ),
            "fin_length": _linear(
                values[2],
                cfg.fin_length_min,
                cfg.fin_length_max,
            ),
            "n_fins": choices[candidate_id % len(choices)],
            "gap_to_thickness_ratio": _linear(
                values[3],
                cfg.fin_gap_ratio_min,
                cfg.fin_gap_ratio_max,
            ),
            "taper_ratio": _linear(
                values[4],
                cfg.fin_taper_ratio_min,
                cfg.fin_taper_ratio_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = values[5:8]
    else:
        raise ValueError(f"Unsupported OOD family: {family!r}")

    rotation = _linear(
        rotation_u,
        cfg.geometry_rotation_min,
        cfg.geometry_rotation_max,
    )
    return parameters, rotation, float(centre_x_u), float(centre_y_u)


def _place_inside_domain(
    cfg,
    family,
    parameters,
    rotation,
    centre_x_u,
    centre_y_u,
    candidate_id,
    seed,
):
    provisional = GeometryCase(
        family=family,
        cx=0.0,
        cy=0.0,
        rotation=rotation,
        parameters=parameters,
        candidate_id=candidate_id,
        seed=seed,
        group="ood",
    )
    canonical = build_canonical_ood_boundary(
        provisional,
        curve_segments=cfg.curved_boundary_segments,
    )
    rotated = transform_points(canonical, 0.0, 0.0, rotation=rotation)
    xmin, ymin = rotated.min(axis=0)
    xmax, ymax = rotated.max(axis=0)
    clearance = float(cfg.geometry_minimum_clearance)
    cx_min = clearance - xmin
    cx_max = float(cfg.Lx) - clearance - xmax
    cy_min = clearance - ymin
    cy_max = float(cfg.Ly) - clearance - ymax
    if cx_min > cx_max or cy_min > cy_max:
        raise ValueError("OOD geometry cannot fit inside the configured domain.")

    if cfg.random_position:
        cx = _linear(centre_x_u, cx_min, cx_max)
        cy = _linear(centre_y_u, cy_min, cy_max)
    else:
        cx = float(cfg.cx_fixed)
        cy = float(cfg.cy_fixed)
    return GeometryCase(
        family=family,
        cx=cx,
        cy=cy,
        rotation=rotation,
        parameters=parameters,
        candidate_id=candidate_id,
        seed=seed,
        group="ood",
    )


def build_ood_family_case_plan(cfg, family, n_cases, role="ood"):
    """Build a deterministic oversampled OOD case plan."""
    family = canonical_geometry_name(family)
    if family not in OOD_GEOMETRY_FAMILIES:
        raise ValueError(f"{family!r} is not an OOD family.")
    if int(n_cases) <= 0:
        raise ValueError("n_cases must be positive.")

    family_seed = int(cfg.seed) + int(cfg.geometry_seed_offsets[family]) + ROLE_SEED_OFFSETS["ood"]
    samples = parameter_candidates(int(n_cases), _parameter_count(family), "ood",
                                   family_seed, cfg.ood_candidate_multiplier)

    cases = []
    for candidate_id, values in enumerate(samples):
        parameters, rotation, centre_x_u, centre_y_u = _sample_parameters(
            cfg,
            family,
            values,
            candidate_id,
        )
        try:
            case = _place_inside_domain(
                cfg,
                family,
                parameters,
                rotation,
                centre_x_u,
                centre_y_u,
                candidate_id,
                family_seed,
            )
        except ValueError:
            continue
        cases.append(case)
    return cases


def generate_ood_family_dataset(cfg, family, n_cases=None):
    from src.data.generation import generate_family
    family = canonical_geometry_name(family)
    if family not in OOD_GEOMETRY_FAMILIES:
        raise ValueError(f"Not an OOD family: {family}")
    n_cases = int(cfg.ood_cases_per_family[family] if n_cases is None else n_cases)
    return generate_family(cfg, family, cfg.ood_dataset_path(family, n_cases), {"ood": n_cases},
                           build_ood_family_case_plan, sample_ood_geometry_case, ood=True)


def generate_selected_ood_datasets(cfg):
    import numpy as np
    from src.data.design import assert_unique_geometries
    # Keep OOD disjoint from all generated in-distribution cases as well.
    with np.load(cfg.dataset_path, allow_pickle=True) as data:
        assert_unique_geometries(data['geometry_keys'], data['roles'], data['metadata_json'])
        cfg._generation_seen = set(data['geometry_keys'].tolist())
    try:
        return {family: generate_ood_family_dataset(cfg, family)
                for family in cfg.generate_ood_families if cfg.ood_cases_per_family[family] > 0}
    finally:
        del cfg._generation_seen
