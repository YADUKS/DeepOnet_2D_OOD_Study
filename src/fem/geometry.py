"""
Geometry, SDF and NGSolve mesh creation utilities.
"""

import numpy as np
from ngsolve import Mesh
from netgen.geom2d import SplineGeometry

from src.fem.geometry_spec import (
    GeometryCase,
    ensure_counterclockwise,
    signed_distance_to_polygon,
    transform_points,
    validate_boundary,
)
from src.fem.simple_geometries import build_canonical_simple_boundary
from src.fem.ood_geometries import build_canonical_ood_boundary


def prepare_polygon_boundary(cfg, canonical_points, case: GeometryCase):
    """
    Transform and validate a canonical polygon for later mesh/SDF creation.

    The existing circle/rectangle workflow does not call this yet; it becomes
    the shared entry point as geometry families are added in Change 2.
    """
    points = transform_points(
        canonical_points,
        cx=case.cx,
        cy=case.cy,
        rotation=case.rotation,
        scale=case.scale,
    )
    points = ensure_counterclockwise(points)
    diagnostics = validate_boundary(
        points,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        minimum_clearance=cfg.geometry_minimum_clearance,
        minimum_area=cfg.geometry_minimum_area,
        minimum_edge_length=cfg.geometry_minimum_edge_length,
    )
    return points, diagnostics


def sdf_polygon(x, y, boundary_points):
    """SDF for a validated polygon, negative inside the hollow region."""
    return signed_distance_to_polygon(x, y, boundary_points)


def build_polygon_mesh(cfg, boundary_points):
    """Build an NGSolve mesh for a general single closed hollow boundary."""
    points = ensure_counterclockwise(boundary_points)
    validate_boundary(
        points,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        minimum_clearance=cfg.geometry_minimum_clearance,
        minimum_area=cfg.geometry_minimum_area,
        minimum_edge_length=cfg.geometry_minimum_edge_length,
    )

    geo = SplineGeometry()
    geo.AddRectangle(
        (0.0, 0.0),
        (cfg.Lx, cfg.Ly),
        leftdomain=1,
        rightdomain=0,
        bcs=["bottom", "right", "top", "left"],
    )
    point_ids = [geo.AppendPoint(float(x), float(y)) for x, y in points]
    for index, start_id in enumerate(point_ids):
        end_id = point_ids[(index + 1) % len(point_ids)]
        geo.Append(
            ["line", start_id, end_id],
            leftdomain=0,
            rightdomain=1,
            bc="hole",
        )

    ngmesh = geo.GenerateMesh(maxh=cfg.maxh)
    return Mesh(ngmesh)


def prepare_simple_geometry(cfg, case: GeometryCase):
    """Return the transformed boundary and diagnostics for a simple case."""
    canonical_points = build_canonical_simple_boundary(
        case,
        curve_segments=cfg.curved_boundary_segments,
    )
    return prepare_polygon_boundary(cfg, canonical_points, case)


def build_simple_mesh_and_sdf(cfg, case: GeometryCase, Xg, Yg):
    """
    Build the FE mesh and SDF from one shared simple-geometry boundary.

    This is the entry point that the per-family generator will use in Change 3.
    """
    boundary_points, diagnostics = prepare_simple_geometry(cfg, case)
    mesh = build_polygon_mesh(cfg, boundary_points)
    sdf = sdf_polygon(Xg, Yg, boundary_points).astype(np.float32)
    return mesh, sdf, boundary_points, diagnostics


def prepare_ood_geometry(cfg, case: GeometryCase):
    """Return the transformed boundary and diagnostics for an OOD case."""
    canonical_points = build_canonical_ood_boundary(
        case,
        curve_segments=cfg.curved_boundary_segments,
    )
    return prepare_polygon_boundary(cfg, canonical_points, case)


def build_ood_mesh_and_sdf(cfg, case: GeometryCase, Xg, Yg):
    """Build an OOD FE mesh and SDF from one shared validated boundary."""
    boundary_points, diagnostics = prepare_ood_geometry(cfg, case)
    mesh = build_polygon_mesh(cfg, boundary_points)
    sdf = sdf_polygon(Xg, Yg, boundary_points).astype(np.float32)
    return mesh, sdf, boundary_points, diagnostics


def sdf_circle(x, y, cx, cy, D):
    r = D / 2.0
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r


def sdf_rectangle(x, y, cx, cy, W, H):
    dx = np.abs(x - cx) - W / 2.0
    dy = np.abs(y - cy) - H / 2.0
    dx_pos = np.maximum(dx, 0.0)
    dy_pos = np.maximum(dy, 0.0)
    outside_dist = np.sqrt(dx_pos ** 2 + dy_pos ** 2)
    inside_dist = np.minimum(np.maximum(dx, dy), 0.0)
    return outside_dist + inside_dist


def sdf_rounded_rectangle(x, y, cx, cy, W, H, R):
    radius = min(float(R), 0.5 * min(float(W), float(H)) - 1e-12)
    qx = np.abs(x - cx) - (W / 2.0 - radius)
    qy = np.abs(y - cy) - (H / 2.0 - radius)
    qx_pos = np.maximum(qx, 0.0)
    qy_pos = np.maximum(qy, 0.0)
    outside_dist = np.sqrt(qx_pos ** 2 + qy_pos ** 2)
    inside_dist = np.minimum(np.maximum(qx, qy), 0.0)
    return outside_dist + inside_dist - radius


def choose_circle_center(cfg, D, rng):
    if not cfg.random_position:
        return cfg.cx_fixed, cfg.cy_fixed
    r = D / 2.0
    cx = rng.uniform(r, cfg.Lx - r)
    cy = rng.uniform(r, cfg.Ly - r)
    return cx, cy


def choose_rect_center(cfg, W, H, rng):
    if not cfg.random_position:
        return cfg.cx_fixed, cfg.cy_fixed
    cx = rng.uniform(W / 2.0, cfg.Lx - W / 2.0)
    cy = rng.uniform(H / 2.0, cfg.Ly - H / 2.0)
    return cx, cy


def build_circle_mesh(cfg, D, cx, cy):
    geo = SplineGeometry()
    geo.AddRectangle(
        (0.0, 0.0),
        (cfg.Lx, cfg.Ly),
        leftdomain=1,
        rightdomain=0,
        bcs=["bottom", "right", "top", "left"],
    )
    geo.AddCircle((cx, cy), D / 2.0, leftdomain=0, rightdomain=1, bc="hole")
    ngmesh = geo.GenerateMesh(maxh=cfg.maxh)
    return Mesh(ngmesh)


def build_rectangle_mesh(cfg, W, H, cx, cy):
    geo = SplineGeometry()
    geo.AddRectangle(
        (0.0, 0.0),
        (cfg.Lx, cfg.Ly),
        leftdomain=1,
        rightdomain=0,
        bcs=["bottom", "right", "top", "left"],
    )

    x1, y1 = cx - W / 2.0, cy - H / 2.0
    x2, y2 = cx + W / 2.0, cy - H / 2.0
    x3, y3 = cx + W / 2.0, cy + H / 2.0
    x4, y4 = cx - W / 2.0, cy + H / 2.0

    p1 = geo.AppendPoint(x1, y1)
    p2 = geo.AppendPoint(x2, y2)
    p3 = geo.AppendPoint(x3, y3)
    p4 = geo.AppendPoint(x4, y4)

    geo.Append(["line", p1, p2], leftdomain=0, rightdomain=1, bc="hole")
    geo.Append(["line", p2, p3], leftdomain=0, rightdomain=1, bc="hole")
    geo.Append(["line", p3, p4], leftdomain=0, rightdomain=1, bc="hole")
    geo.Append(["line", p4, p1], leftdomain=0, rightdomain=1, bc="hole")

    ngmesh = geo.GenerateMesh(maxh=cfg.maxh)
    return Mesh(ngmesh)


def rounded_rectangle_boundary_points(cx, cy, W, H, R, n_corner_segments):
    radius = min(float(R), 0.5 * min(float(W), float(H)) - 1e-12)
    a = W / 2.0 - radius
    b = H / 2.0 - radius
    nseg = max(int(n_corner_segments), 2)

    corner_specs = [
        (cx + a, cy - b, -0.5 * np.pi, 0.0),
        (cx + a, cy + b, 0.0, 0.5 * np.pi),
        (cx - a, cy + b, 0.5 * np.pi, np.pi),
        (cx - a, cy - b, np.pi, 1.5 * np.pi),
    ]

    points = []
    for x0, y0, theta0, theta1 in corner_specs:
        theta = np.linspace(theta0, theta1, nseg + 1)
        for t in theta:
            pt = (x0 + radius * np.cos(t), y0 + radius * np.sin(t))
            if not points or np.hypot(pt[0] - points[-1][0], pt[1] - points[-1][1]) > 1e-14:
                points.append(pt)

    if len(points) > 1 and np.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 1e-14:
        points.pop()

    return points


def build_rounded_rectangle_mesh(cfg, W, H, R, cx, cy):
    geo = SplineGeometry()
    geo.AddRectangle(
        (0.0, 0.0),
        (cfg.Lx, cfg.Ly),
        leftdomain=1,
        rightdomain=0,
        bcs=["bottom", "right", "top", "left"],
    )

    points = rounded_rectangle_boundary_points(
        cx,
        cy,
        W,
        H,
        R,
        getattr(cfg, "rounded_corner_segments", 8),
    )
    point_ids = [geo.AppendPoint(x, y) for x, y in points]

    for i, p_start in enumerate(point_ids):
        p_end = point_ids[(i + 1) % len(point_ids)]
        geo.Append(["line", p_start, p_end], leftdomain=0, rightdomain=1, bc="hole")

    ngmesh = geo.GenerateMesh(maxh=cfg.maxh)
    return Mesh(ngmesh)


def build_mesh_and_sdf(cfg, shape, p1, p2, cx, cy, Xg, Yg, p3=0.0):
    if shape == "circle":
        mesh = build_circle_mesh(cfg, p1, cx, cy)
        sdf = sdf_circle(Xg, Yg, cx, cy, p1)
    elif shape == "rect":
        mesh = build_rectangle_mesh(cfg, p1, p2, cx, cy)
        sdf = sdf_rectangle(Xg, Yg, cx, cy, p1, p2)
    elif shape == "rounded_rect":
        mesh = build_rounded_rectangle_mesh(cfg, p1, p2, p3, cx, cy)
        sdf = sdf_rounded_rectangle(Xg, Yg, cx, cy, p1, p2, p3)
    else:
        raise ValueError("Unknown shape. Use 'circle', 'rect', or 'rounded_rect'.")
    return mesh, sdf.astype(np.float32)
