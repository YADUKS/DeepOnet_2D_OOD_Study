"""
Common geometry descriptions and geometry-independent helper functions.

This module intentionally has no NGSolve dependency.  Geometry sampling,
dataset composition, plotting, and OOD evaluation can therefore share one
case representation without importing the finite-element stack.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


from .registry import TRAINING_NAMES
TRAINING_GEOMETRY_FAMILIES = TRAINING_NAMES

OOD_GEOMETRY_FAMILIES = (
    "star",
    "aircraft",
    "leaf",
    "fin_like",
)

GEOMETRY_ALIASES = {
    "rect": "rectangle",
    "polygon": "regular_polygon",

    "t": "t_shape",
    "tshape": "t_shape",

    "cross": "cross_shape",
    "plus": "cross_shape",
    "plus_shape": "cross_shape",
    
    "fin": "fin_like",
    "engineering_fin": "fin_like",
    "engineering_fin_like": "fin_like",
}


def canonical_geometry_name(name):
    """Return the canonical lower-case name for a geometry family."""
    canonical = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return GEOMETRY_ALIASES.get(canonical, canonical)


@dataclass(frozen=True)
class GeometryCase:
    """
    Reproducible description of one hollow geometry.

    Common transformations are kept separate from family-specific parameters.
    The network does not consume this metadata; its geometry input remains SDF.
    """

    family: str
    cx: float
    cy: float
    rotation: float = 0.0
    scale: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    case_id: int = -1
    candidate_id: int = -1
    seed: int = 0
    group: str = "training"

    def __post_init__(self):
        family = canonical_geometry_name(self.family)
        group = str(self.group).strip().lower()

        if family not in TRAINING_GEOMETRY_FAMILIES + OOD_GEOMETRY_FAMILIES:
            raise ValueError(f"Unknown geometry family: {self.family!r}")
        if group not in ("training", "ood"):
            raise ValueError("Geometry group must be 'training' or 'ood'.")
        if not np.isfinite([self.cx, self.cy, self.rotation, self.scale]).all():
            raise ValueError("Geometry transform values must be finite.")
        if self.scale <= 0.0:
            raise ValueError("Geometry scale must be positive.")

        object.__setattr__(self, "family", family)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_record(self):
        """Return a JSON-compatible metadata record."""
        return {
            "family": self.family,
            "group": self.group,
            "case_id": int(self.case_id),
            "candidate_id": int(self.candidate_id),
            "seed": int(self.seed),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rotation": float(self.rotation),
            "scale": float(self.scale),
            "parameters": dict(self.parameters),
        }


def as_point_array(points):
    """Validate and return boundary points as an ``[N, 2]`` float array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Boundary points must have shape [N, 2].")
    if array.shape[0] < 3:
        raise ValueError("A closed boundary requires at least three points.")
    if not np.isfinite(array).all():
        raise ValueError("Boundary points must be finite.")

    if np.linalg.norm(array[0] - array[-1]) <= 1e-14:
        array = array[:-1]
    if array.shape[0] < 3:
        raise ValueError("A closed boundary requires three distinct points.")
    return array


def rotate_points(points, angle):
    """Rotate boundary points counter-clockwise around the origin."""
    points = as_point_array(points)
    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    rotation_matrix = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return points @ rotation_matrix.T


def transform_points(points, cx, cy, rotation=0.0, scale=1.0):
    """Apply uniform scale, rotation, and translation to canonical points."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Geometry scale must be a positive finite number.")
    transformed = rotate_points(as_point_array(points) * float(scale), rotation)
    transformed[:, 0] += float(cx)
    transformed[:, 1] += float(cy)
    return transformed


def polygon_signed_area(points):
    """Return signed polygon area; positive means counter-clockwise order."""
    points = as_point_array(points)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ensure_counterclockwise(points):
    """Return boundary points in counter-clockwise order."""
    points = as_point_array(points)
    if polygon_signed_area(points) < 0.0:
        return points[::-1].copy()
    return points.copy()


def polygon_perimeter(points):
    """Return the perimeter of a closed polygonal boundary."""
    points = as_point_array(points)
    edges = np.roll(points, -1, axis=0) - points
    return float(np.linalg.norm(edges, axis=1).sum())


def polygon_bounds(points):
    """Return ``(xmin, xmax, ymin, ymax)``."""
    points = as_point_array(points)
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def minimum_domain_clearance(points, Lx, Ly):
    """Return the minimum vertex clearance from the rectangular outer domain."""
    xmin, xmax, ymin, ymax = polygon_bounds(points)
    return float(min(xmin, float(Lx) - xmax, ymin, float(Ly) - ymax))


def _orientation(a, b, c):
    u, v = b - a, c - a
    return float(u[0] * v[1] - u[1] * v[0])


def _segments_intersect(a, b, c, d, tolerance=1e-14):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    return (
        ((o1 > tolerance and o2 < -tolerance) or
         (o1 < -tolerance and o2 > tolerance))
        and
        ((o3 > tolerance and o4 < -tolerance) or
         (o3 < -tolerance and o4 > tolerance))
    )


def polygon_is_simple(points):
    """Return ``False`` when non-adjacent polygon edges intersect."""
    points = as_point_array(points)
    npoints = len(points)

    for i in range(npoints):
        a = points[i]
        b = points[(i + 1) % npoints]
        for j in range(i + 1, npoints):
            if j == i or (j + 1) % npoints == i or (i + 1) % npoints == j:
                continue
            c = points[j]
            d = points[(j + 1) % npoints]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def validate_boundary(
    points,
    Lx,
    Ly,
    minimum_clearance=0.0,
    minimum_area=0.0,
    minimum_edge_length=0.0,
):
    """
    Validate a polygonal hollow boundary and return diagnostic measurements.

    Raises ``ValueError`` with a specific rejection reason when invalid.
    """
    points = as_point_array(points)
    edges = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edges, axis=1)
    area = abs(polygon_signed_area(points))
    clearance = minimum_domain_clearance(points, Lx, Ly)

    if np.any(edge_lengths <= float(minimum_edge_length)):
        raise ValueError("Geometry contains an edge below the minimum length.")
    if area <= float(minimum_area):
        raise ValueError("Geometry area is below the configured minimum.")
    if not polygon_is_simple(points):
        raise ValueError("Geometry boundary is self-intersecting.")
    if clearance < float(minimum_clearance):
        raise ValueError("Geometry violates the outer-domain clearance.")

    return {
        "area": float(area),
        "perimeter": polygon_perimeter(points),
        "minimum_edge_length": float(edge_lengths.min()),
        "domain_clearance": float(clearance),
        "bounds": polygon_bounds(points),
    }


def signed_distance_to_polygon(x, y, points):
    """
    Compute signed distance to a polygon.

    The sign convention matches the existing SDF functions: negative inside
    the hollow region, zero on its boundary, and positive in the material.
    """
    points = as_point_array(points)
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    query_x = x_array.ravel()
    query_y = y_array.ravel()
    minimum_distance_squared = np.full(query_x.shape, np.inf)
    inside = np.zeros(query_x.shape, dtype=bool)

    for start, end in zip(points, np.roll(points, -1, axis=0)):
        edge = end - start
        edge_length_squared = float(np.dot(edge, edge))
        if edge_length_squared <= 1e-28:
            raise ValueError("Geometry boundary contains a zero-length edge.")

        relative_x = query_x - start[0]
        relative_y = query_y - start[1]
        projection = np.clip(
            (relative_x * edge[0] + relative_y * edge[1]) / edge_length_squared,
            0.0,
            1.0,
        )
        closest_x = start[0] + projection * edge[0]
        closest_y = start[1] + projection * edge[1]
        distance_squared = (
            (query_x - closest_x) ** 2 + (query_y - closest_y) ** 2
        )
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            distance_squared,
        )

        crosses_scanline = (start[1] > query_y) != (end[1] > query_y)
        intersection_x = (
            edge[0] * (query_y - start[1]) / (edge[1] + 1e-300)
            + start[0]
        )
        inside ^= crosses_scanline & (query_x < intersection_x)

    distance = np.sqrt(minimum_distance_squared)
    distance[inside] *= -1.0
    return distance.reshape(x_array.shape)
