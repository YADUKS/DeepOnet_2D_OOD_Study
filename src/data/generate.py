"""
Generate deterministic, independently reusable datasets for simple geometries.

Each family is written to its own directory and file.  No family is combined
here; selecting and composing families for training is handled separately.
"""


from src.fem.geometry_spec import (
    GeometryCase,
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
    transform_points,
)
from src.fem.sampling import sample_geometry_case
from src.fem.simple_geometries import build_canonical_simple_boundary


DATASET_SCHEMA_VERSION = 2


from src.data.design import parameter_candidates, ROLE_SEED_OFFSETS


def _linear(unit_value, lower, upper):
    return float(lower) + float(unit_value) * (float(upper) - float(lower))


def _family_parameter_count(family):
    from src.fem.registry import dimensions
    return dimensions(family)


def _sample_family_parameters(cfg, family, unit_values, candidate_id):
    """Map one unit-cube row to family parameters, rotation, and centre coordinates."""
    family = canonical_geometry_name(family)
    from src.fem.registry import custom_shape
    spec = custom_shape(family)
    if spec is not None:
        return spec["map_parameters"](cfg, unit_values, candidate_id)

    if family == "circle":
        parameters = {
            "diameter": _linear(unit_values[0], cfg.D_min, cfg.D_max),
        }
        rotation = 0.0
        centre_x_u, centre_y_u = unit_values[1:3]
    elif family == "ellipse":
        major_axis = _linear(
            unit_values[0],
            cfg.ellipse_major_axis_min,
            cfg.ellipse_major_axis_max,
        )
        aspect_ratio = _linear(
            unit_values[1],
            cfg.ellipse_aspect_ratio_min,
            cfg.ellipse_aspect_ratio_max,
        )
        parameters = {
            "major_axis": major_axis,
            "minor_axis": major_axis / aspect_ratio,
            "aspect_ratio": aspect_ratio,
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[2:5]
    elif family == "rectangle":
        parameters = {
            "width": _linear(unit_values[0], cfg.W_min, cfg.W_max),
            "height": _linear(unit_values[1], cfg.H_min, cfg.H_max),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[2:5]
    elif family == "triangle":
        parameters = {
            "base": _linear(
                unit_values[0],
                cfg.triangle_base_min,
                cfg.triangle_base_max,
            ),
            "height": _linear(
                unit_values[1],
                cfg.triangle_height_min,
                cfg.triangle_height_max,
            ),
            "skew": _linear(
                unit_values[2],
                cfg.triangle_skew_min,
                cfg.triangle_skew_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[3:6]
    elif family == "rhombus":
        parameters = {
            "diagonal_x": _linear(
                unit_values[0],
                cfg.rhombus_diagonal_x_min,
                cfg.rhombus_diagonal_x_max,
            ),
            "diagonal_y": _linear(
                unit_values[1],
                cfg.rhombus_diagonal_y_min,
                cfg.rhombus_diagonal_y_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[2:5]
    elif family == "trapezoid":
        bottom_width = _linear(
            unit_values[0],
            cfg.trapezoid_bottom_width_min,
            cfg.trapezoid_bottom_width_max,
        )
        top_ratio = _linear(
            unit_values[1],
            cfg.trapezoid_top_ratio_min,
            cfg.trapezoid_top_ratio_max,
        )
        parameters = {
            "bottom_width": bottom_width,
            "top_width": bottom_width * top_ratio,
            "top_ratio": top_ratio,
            "height": _linear(
                unit_values[2],
                cfg.trapezoid_height_min,
                cfg.trapezoid_height_max,
            ),
            "top_offset": bottom_width * _linear(
                unit_values[3],
                cfg.trapezoid_top_offset_fraction_min,
                cfg.trapezoid_top_offset_fraction_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[4:7]
    elif family == "regular_polygon":
        side_choices = tuple(int(value) for value in cfg.regular_polygon_sides)
        sides = side_choices[int(candidate_id) % len(side_choices)]
        parameters = {
            "sides": sides,
            "circumradius": _linear(
                unit_values[0],
                cfg.regular_polygon_radius_min,
                cfg.regular_polygon_radius_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[2:5]
    elif family == "t_shape":
        parameters = {
            "top_width": _linear(
                unit_values[0],
                cfg.t_shape_top_width_min,
                cfg.t_shape_top_width_max,
            ),
            "overall_height": _linear(
                unit_values[1],
                cfg.t_shape_height_min,
                cfg.t_shape_height_max,
            ),
            "stem_width_ratio": _linear(
                unit_values[2],
                cfg.t_shape_stem_width_ratio_min,
                cfg.t_shape_stem_width_ratio_max,
            ),
            "top_thickness_ratio": _linear(
                unit_values[3],
                cfg.t_shape_top_thickness_ratio_min,
                cfg.t_shape_top_thickness_ratio_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[4:7]
    elif family == "cross_shape":
        parameters = {
            "overall_width": _linear(
                unit_values[0],
                cfg.cross_shape_width_min,
                cfg.cross_shape_width_max,
            ),
            "overall_height": _linear(
                unit_values[1],
                cfg.cross_shape_height_min,
                cfg.cross_shape_height_max,
            ),
            "vertical_width_ratio": _linear(
                unit_values[2],
                cfg.cross_shape_vertical_width_ratio_min,
                cfg.cross_shape_vertical_width_ratio_max,
            ),
            "horizontal_thickness_ratio": _linear(
                unit_values[3],
                cfg.cross_shape_horizontal_thickness_ratio_min,
                cfg.cross_shape_horizontal_thickness_ratio_max,
            ),
        }
        rotation_u, centre_x_u, centre_y_u = unit_values[4:7]
    else:
        raise ValueError(f"Unsupported simple geometry family: {family!r}")

    if family != "circle":
        rotation = _linear(
            rotation_u,
            cfg.geometry_rotation_min,
            cfg.geometry_rotation_max,
        )
    return parameters, rotation, float(centre_x_u), float(centre_y_u)


def _place_case_inside_domain(
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
        group="training",
    )
    canonical = build_canonical_simple_boundary(
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
        raise ValueError("Geometry cannot fit inside the configured domain.")

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
        group="training",
    )


def build_simple_family_case_plan(cfg, family, n_cases, role="train"):
    """Build a deterministic oversampled case plan for one family."""
    family = canonical_geometry_name(family)
    if family not in TRAINING_GEOMETRY_FAMILIES:
        raise ValueError(f"{family!r} is not a simple training family.")
    if int(n_cases) <= 0:
        raise ValueError("n_cases must be positive.")

    family_seed = int(cfg.seed) + int(cfg.geometry_seed_offsets[family]) + ROLE_SEED_OFFSETS[role]
    samples = parameter_candidates(int(n_cases), _family_parameter_count(family), role,
                                   family_seed, cfg.geometry_candidate_multiplier)

    cases = []
    for candidate_id, unit_values in enumerate(samples):
        parameters, rotation, centre_x_u, centre_y_u = _sample_family_parameters(
            cfg,
            family,
            unit_values,
            candidate_id,
        )
        try:
            case = _place_case_inside_domain(
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


def generate_simple_family_dataset(cfg, family, n_cases=None):
    """Generate a Halton pool and independent, fixed LHS holdouts."""
    from src.data.generation import generate_family
    family = canonical_geometry_name(family)
    if family not in TRAINING_GEOMETRY_FAMILIES:
        raise ValueError(f"Not a simple training family: {family}")
    n_cases = int(cfg.geometry_cases_per_family[family] if n_cases is None else n_cases)
    # Freeze holdouts first. Training skips any exact holdout duplicate, so
    # increasing the pool cannot change validation/test membership.
    quotas = {"val": int(cfg.validation_cases_per_family[family]),
              "test": int(cfg.test_cases_per_family[family]), "train": n_cases}
    return generate_family(cfg, family, cfg.geometry_dataset_path(family, n_cases), quotas,
                           build_simple_family_case_plan, sample_geometry_case)


def generate_selected_simple_geometry_datasets(cfg):
    cfg._generation_seen = set()
    try:
        return {family: generate_simple_family_dataset(cfg, family)
                for family in cfg.generate_geometry_families if cfg.geometry_cases_per_family[family] > 0}
    finally:
        del cfg._generation_seen
