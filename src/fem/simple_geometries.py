"""
Boundary constructors for the simple hollow geometries used for training.

Every constructor creates a canonical boundary around the origin.  The common
``GeometryCase`` scale, rotation, and translation are applied afterwards so
the NGSolve mesh and Cartesian-grid SDF use exactly the same final boundary.
"""

import numpy as np

from src.fem.geometry_spec import (
    GeometryCase,
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)


def _required_float(parameters, name):
    try:
        value = float(parameters[name])
    except KeyError as exc:
        raise ValueError(f"Missing required geometry parameter: {name!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Geometry parameter {name!r} must be finite.")
    return value


def _positive_float(parameters, name):
    value = _required_float(parameters, name)
    if value <= 0.0:
        raise ValueError(f"Geometry parameter {name!r} must be positive.")
    return value


def circle_boundary(parameters, n_segments):
    diameter = _positive_float(parameters, "diameter")
    radius = 0.5 * diameter
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_segments), endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def ellipse_boundary(parameters, n_segments):
    major_axis = _positive_float(parameters, "major_axis")
    minor_axis = _positive_float(parameters, "minor_axis")
    if minor_axis > major_axis:
        raise ValueError("Ellipse minor_axis cannot exceed major_axis.")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_segments), endpoint=False)
    return np.column_stack(
        (
            0.5 * major_axis * np.cos(angles),
            0.5 * minor_axis * np.sin(angles),
        )
    )


def rectangle_boundary(parameters):
    width = _positive_float(parameters, "width")
    height = _positive_float(parameters, "height")
    half_width = 0.5 * width
    half_height = 0.5 * height
    return np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )


def triangle_boundary(parameters):
    base = _positive_float(parameters, "base")
    height = _positive_float(parameters, "height")
    skew = float(parameters.get("skew", 0.0))
    if not np.isfinite(skew) or not -0.45 <= skew <= 0.45:
        raise ValueError("Triangle skew must be between -0.45 and 0.45.")

    apex_x = skew * base
    points = np.array(
        [
            [-0.5 * base, 0.0],
            [0.5 * base, 0.0],
            [apex_x, height],
        ],
        dtype=np.float64,
    )
    return points - points.mean(axis=0)


def rhombus_boundary(parameters):
    diagonal_x = _positive_float(parameters, "diagonal_x")
    diagonal_y = _positive_float(parameters, "diagonal_y")
    return np.array(
        [
            [0.5 * diagonal_x, 0.0],
            [0.0, 0.5 * diagonal_y],
            [-0.5 * diagonal_x, 0.0],
            [0.0, -0.5 * diagonal_y],
        ],
        dtype=np.float64,
    )


def trapezoid_boundary(parameters):
    bottom_width = _positive_float(parameters, "bottom_width")
    top_width = _positive_float(parameters, "top_width")
    height = _positive_float(parameters, "height")
    top_offset = float(parameters.get("top_offset", 0.0))
    if not np.isfinite(top_offset):
        raise ValueError("Trapezoid top_offset must be finite.")

    points = np.array(
        [
            [-0.5 * bottom_width, 0.0],
            [0.5 * bottom_width, 0.0],
            [top_offset + 0.5 * top_width, height],
            [top_offset - 0.5 * top_width, height],
        ],
        dtype=np.float64,
    )
    return points - points.mean(axis=0)

def t_shape_boundary(parameters):
    """
    Construct a centred T-shaped polygon.

    The canonical T-shape is aligned vertically:
    - the horizontal bar is at the top;
    - the vertical stem extends downward;
    - rotation and translation are applied later.
    """
    top_width = _positive_float(parameters, "top_width")
    overall_height= _positive_float(parameters, "overall_height")

    stem_width_ratio=_required_float(
        parameters, 
        "stem_width_ratio",
    )

    top_thickness_ratio = _required_float(
        parameters,
        "top_thickness_ratio",
    )
    if not 0.10 <= stem_width_ratio <= 0.80:
        raise ValueError(
            "T-shape stem_width_ratio must lie between 0.10 and 0.80."
        )

    if not 0.10 <= top_thickness_ratio <= 0.60:
        raise ValueError(
            "T-shape top_thickness_ratio must lie between 0.10 and 0.60."
        )
    stem_width = stem_width_ratio * top_width
    top_thickness = top_thickness_ratio * overall_height

    half_top_width = 0.5 * top_width
    half_stem_width = 0.5 * stem_width
    half_height = 0.5 * overall_height

    top_y = half_height
    bar_bottom_y = top_y - top_thickness
    bottom_y = -half_height

    return np.array(
        [   
            [-half_stem_width, bottom_y],
            [ half_stem_width, bottom_y],
            [ half_stem_width, bar_bottom_y],
            [ half_top_width,  bar_bottom_y],
            [ half_top_width,  top_y],
            [-half_top_width,  top_y],
            [-half_top_width,  bar_bottom_y],
            [-half_stem_width, bar_bottom_y],
        ],
        dtype=np.float64,
    )

def cross_shape_boundary(parameters):
    """
    Construct a centred cross-shaped polygon.

    The canonical cross consists of:
    - one vertical rectangular arm;
    - one horizontal rectangular arm;
    - a centred intersection.

    Rotation and translation are applied later.
    """

    overall_width = _positive_float(parameters, "overall_width",)
    overall_height = _positive_float(parameters, "overall_height",)
    vertical_width_ratio=_required_float(parameters, "vertical_width_ratio",)
    horizontal_thickness_ratio = _required_float(
        parameters,
        "horizontal_thickness_ratio",
    )

    if not 0.10 <= vertical_width_ratio <= 0.80:
        raise ValueError(
            "Cross-shape vertical_width_ratio must lie between "
            "0.10 and 0.80."
        )

    if not 0.10 <= horizontal_thickness_ratio <= 0.80:
        raise ValueError(
            "Cross-shape horizontal_thickness_ratio must lie between "
            "0.10 and 0.80."
        )
    vertical_width = vertical_width_ratio * overall_width

    horizontal_thickness = (
        horizontal_thickness_ratio * overall_height
    )
    half_width = 0.5 * overall_width
    half_height = 0.5 * overall_height
    half_vertical_width = 0.5 * vertical_width
    half_horizontal_thickness = 0.5 * horizontal_thickness

    return np.array(
        [
            [-half_vertical_width, -half_height],
            [ half_vertical_width, -half_height],

            [ half_vertical_width, -half_horizontal_thickness],
            [ half_width,          -half_horizontal_thickness],

            [ half_width,           half_horizontal_thickness],
            [ half_vertical_width,  half_horizontal_thickness],

            [ half_vertical_width,  half_height],
            [-half_vertical_width,  half_height],

            [-half_vertical_width,  half_horizontal_thickness],
            [-half_width,           half_horizontal_thickness],

            [-half_width,          -half_horizontal_thickness],
            [-half_vertical_width, -half_horizontal_thickness],
        ],
        dtype=np.float64,
    )

def regular_polygon_boundary(parameters):
    sides_raw = _required_float(parameters, "sides")
    sides = int(round(sides_raw))
    if not np.isclose(sides_raw, sides) or not 5 <= sides <= 10:
        raise ValueError("Regular polygon sides must be an integer from 5 to 10.")
    circumradius = _positive_float(parameters, "circumradius")
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return np.column_stack(
        (circumradius * np.cos(angles), circumradius * np.sin(angles))
    )


def build_canonical_simple_boundary(case: GeometryCase, curve_segments=96):
    from src.fem.registry import custom_shape
    spec = custom_shape(case.family)
    if spec is not None:
        return np.asarray(spec["boundary"](case.parameters, curve_segments), dtype=np.float64)
    """Construct the untransformed boundary for a simple geometry case."""
    family = canonical_geometry_name(case.family)
    if family not in TRAINING_GEOMETRY_FAMILIES:
        raise ValueError(f"{family!r} is not a simple training geometry.")

    n_segments = int(curve_segments)
    if n_segments < 24:
        raise ValueError("curve_segments must be at least 24.")

    constructors = {
        "circle": lambda: circle_boundary(case.parameters, n_segments),
        "ellipse": lambda: ellipse_boundary(case.parameters, n_segments),
        "rectangle": lambda: rectangle_boundary(case.parameters),
        "triangle": lambda: triangle_boundary(case.parameters),
        "rhombus": lambda: rhombus_boundary(case.parameters),
        "trapezoid": lambda: trapezoid_boundary(case.parameters),
        "regular_polygon": lambda: regular_polygon_boundary(case.parameters),
        "t_shape": lambda: t_shape_boundary(case.parameters),
        "cross_shape": lambda: cross_shape_boundary(case.parameters),
    }
    return constructors[family]()


def representative_simple_cases(cfg):
    """Return one valid centred example of every simple family for smoke tests."""
    common = {
        "cx": 0.5 * cfg.Lx,
        "cy": 0.5 * cfg.Ly,
        "rotation": 0.2,
        "group": "training",
    }
    return (
        GeometryCase("circle", parameters={"diameter": 0.010}, **common),
        GeometryCase(
            "ellipse",
            parameters={"major_axis": 0.012, "minor_axis": 0.007},
            **common,
        ),
        GeometryCase(
            "rectangle",
            parameters={"width": 0.011, "height": 0.007},
            **common,
        ),
        GeometryCase(
            "triangle",
            parameters={"base": 0.011, "height": 0.009, "skew": 0.12},
            **common,
        ),
        GeometryCase(
            "rhombus",
            parameters={"diagonal_x": 0.012, "diagonal_y": 0.008},
            **common,
        ),
        GeometryCase(
            "trapezoid",
            parameters={
                "bottom_width": 0.012,
                "top_width": 0.008,
                "height": 0.008,
                "top_offset": 0.0005,
            },
            **common,
        ),
        GeometryCase(
            "regular_polygon",
            parameters={"sides": 7, "circumradius": 0.0055},
            **common,
        ),
    )
