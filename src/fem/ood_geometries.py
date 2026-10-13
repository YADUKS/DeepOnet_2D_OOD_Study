"""
Boundary constructors for out-of-distribution hollow geometry families.

These families are never included in the simple-geometry training composer.
"""

import numpy as np

from src.fem.geometry_spec import (
    GeometryCase,
    OOD_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)


def _required_float(parameters, name):
    try:
        value = float(parameters[name])
    except KeyError as exc:
        raise ValueError(f"Missing required OOD parameter: {name!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"OOD parameter {name!r} must be finite.")
    return value


def _positive_float(parameters, name):
    value = _required_float(parameters, name)
    if value <= 0.0:
        raise ValueError(f"OOD parameter {name!r} must be positive.")
    return value


def _required_integer(parameters, name, minimum, maximum):
    raw = _required_float(parameters, name)
    value = int(round(raw))
    if not np.isclose(raw, value) or not minimum <= value <= maximum:
        raise ValueError(
            f"OOD parameter {name!r} must be an integer from "
            f"{minimum} to {maximum}."
        )
    return value


def star_boundary(parameters):
    points = _required_integer(parameters, "points", 5, 8)
    outer_radius = _positive_float(parameters, "outer_radius")
    inner_ratio = _required_float(parameters, "inner_ratio")
    if not 0.25 <= inner_ratio <= 0.75:
        raise ValueError("Star inner_ratio must lie between 0.25 and 0.75.")
    radii = np.empty(2 * points, dtype=np.float64)
    radii[0::2] = outer_radius
    radii[1::2] = outer_radius * inner_ratio
    angles = np.linspace(0.0, 2.0 * np.pi, 2 * points, endpoint=False)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def aircraft_boundary(parameters):
    """
    Construct a symmetric, single-boundary top-view aircraft silhouette.

    Length is aligned with canonical y; rotation is applied later.
    """
    length = _positive_float(parameters, "length")
    wingspan = _positive_float(parameters, "wingspan")
    fuselage_ratio = _required_float(parameters, "fuselage_width_ratio")
    tailspan_ratio = _required_float(parameters, "tailspan_ratio")
    wing_sweep = _required_float(parameters, "wing_sweep")
    if not 0.08 <= fuselage_ratio <= 0.30:
        raise ValueError("Aircraft fuselage_width_ratio is outside limits.")
    if not 0.25 <= tailspan_ratio <= 0.70:
        raise ValueError("Aircraft tailspan_ratio is outside limits.")
    if not 0.0 <= wing_sweep <= 0.35:
        raise ValueError("Aircraft wing_sweep is outside limits.")

    half_width = 0.5 * fuselage_ratio * wingspan
    half_span = 0.5 * wingspan
    half_tail_span = 0.5 * tailspan_ratio * wingspan
    nose = 0.5 * length
    tail = -0.5 * length
    wing_tip_y = (0.05 - wing_sweep) * length

    right_half = np.array(
        [
            [0.0, nose],
            [0.65 * half_width, 0.45 * length],
            [half_width, 0.28 * length],
            [half_width, 0.12 * length],
            [half_span, wing_tip_y],
            [half_span, wing_tip_y - 0.10 * length],
            [half_width, -0.02 * length],
            [0.85 * half_width, -0.30 * length],
            [half_tail_span, -0.38 * length],
            [half_tail_span, -0.47 * length],
            [0.55 * half_width, -0.43 * length],
            [0.0, tail],
        ],
        dtype=np.float64,
    )
    left_half = right_half[-2:0:-1].copy()
    left_half[:, 0] *= -1.0
    return np.vstack((right_half, left_half))


def leaf_boundary(parameters, n_segments):
    """Construct a smooth pointed leaf without internal veins or holes."""
    length = _positive_float(parameters, "length")
    width = _positive_float(parameters, "width")
    curvature = _required_float(parameters, "curvature")
    sharpness = _required_float(parameters, "sharpness")
    if not -0.35 <= curvature <= 0.35:
        raise ValueError("Leaf curvature must lie between -0.35 and 0.35.")
    if not 0.65 <= sharpness <= 1.75:
        raise ValueError("Leaf sharpness must lie between 0.65 and 1.75.")

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_segments), endpoint=False)
    x = 0.5 * length * np.cos(theta)
    sine = np.sin(theta)
    y = (
        0.5
        * width
        * np.sign(sine)
        * np.abs(sine) ** sharpness
        * (1.0 + curvature * np.cos(theta))
    )
    return np.column_stack((x, y))


def fin_like_boundary(parameters):
    """
    Construct one connected engineering comb profile.

    A vertical rectangular spine supports horizontal rectangular fins.  The
    number, length, thickness/gap ratio, and length taper can all vary.
    """
    height = _positive_float(parameters, "height")
    spine_width = _positive_float(parameters, "spine_width")
    fin_length = _positive_float(parameters, "fin_length")
    n_fins = _required_integer(parameters, "n_fins", 4, 8)
    gap_ratio = _required_float(parameters, "gap_to_thickness_ratio")
    taper_ratio = _required_float(parameters, "taper_ratio")
    if not 0.35 <= gap_ratio <= 2.5:
        raise ValueError("Fin gap_to_thickness_ratio is outside limits.")
    if not 0.55 <= taper_ratio <= 1.0:
        raise ValueError("Fin taper_ratio is outside limits.")

    thickness = height / (n_fins + (n_fins - 1) * gap_ratio)
    gap = gap_ratio * thickness
    x_left = -0.5 * (spine_width + fin_length)
    x_spine = x_left + spine_width
    y_top = 0.5 * height
    y_bottom = -0.5 * height

    boundary = [[x_left, y_bottom], [x_left, y_top]]
    for fin_index in range(n_fins):
        fraction = fin_index / max(n_fins - 1, 1)
        current_length = fin_length * (
            1.0 - fraction * (1.0 - taper_ratio)
        )
        fin_top = y_top - fin_index * (thickness + gap)
        fin_bottom = fin_top - thickness
        fin_end = x_spine + current_length
        boundary.extend(
            [
                [fin_end, fin_top],
                [fin_end, fin_bottom],
                [x_spine, fin_bottom],
            ]
        )
        if fin_index < n_fins - 1:
            boundary.append([x_spine, fin_bottom - gap])
    boundary.append([x_left, y_bottom])
    boundary = np.asarray(boundary[:-1], dtype=np.float64)
    bounds_min = boundary.min(axis=0)
    bounds_max = boundary.max(axis=0)
    return boundary - 0.5 * (bounds_min + bounds_max)


def build_canonical_ood_boundary(case: GeometryCase, curve_segments=96):
    """Construct an untransformed OOD boundary from a structured case."""
    family = canonical_geometry_name(case.family)
    if family not in OOD_GEOMETRY_FAMILIES:
        raise ValueError(f"{family!r} is not an OOD geometry family.")
    n_segments = int(curve_segments)
    if n_segments < 24:
        raise ValueError("curve_segments must be at least 24.")

    constructors = {
        "star": lambda: star_boundary(case.parameters),
        "aircraft": lambda: aircraft_boundary(case.parameters),
        "leaf": lambda: leaf_boundary(case.parameters, n_segments),
        "fin_like": lambda: fin_like_boundary(case.parameters),
    }
    return constructors[family]()


def representative_ood_cases(cfg):
    """Return one centred representative case from every OOD family."""
    common = {
        "cx": 0.5 * cfg.Lx,
        "cy": 0.5 * cfg.Ly,
        "rotation": 0.1,
        "group": "ood",
    }
    return (
        GeometryCase(
            "star",
            parameters={
                "points": 5,
                "outer_radius": 0.0055,
                "inner_ratio": 0.48,
            },
            **common,
        ),
        GeometryCase(
            "aircraft",
            parameters={
                "length": 0.013,
                "wingspan": 0.012,
                "fuselage_width_ratio": 0.16,
                "tailspan_ratio": 0.42,
                "wing_sweep": 0.14,
            },
            **common,
        ),
        GeometryCase(
            "leaf",
            parameters={
                "length": 0.013,
                "width": 0.007,
                "curvature": 0.12,
                "sharpness": 1.1,
            },
            **common,
        ),
        GeometryCase(
            "fin_like",
            parameters={
                "height": 0.013,
                "spine_width": 0.002,
                "fin_length": 0.008,
                "n_fins": 6,
                "gap_to_thickness_ratio": 1.0,
                "taper_ratio": 0.9,
            },
            **common,
        ),
    )
