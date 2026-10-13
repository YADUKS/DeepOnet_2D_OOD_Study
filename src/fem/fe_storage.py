"""
Extract FE mesh data and store individual FE cases.
"""

import os
import json
import numpy as np


def get_mesh_points_and_triangles(mesh):
    ngm = mesh.ngmesh

    pts = []
    for p in ngm.Points():
        try:
            pts.append([p[0], p[1]])
        except Exception:
            try:
                pts.append([p.p[0], p.p[1]])
            except Exception:
                pts.append([p.x, p.y])
    pts = np.array(pts, dtype=float)

    triangles = []
    for el in ngm.Elements2D():
        vids = []
        if hasattr(el, "vertices"):
            for v in el.vertices:
                nr = getattr(v, "nr", None)
                if nr is None:
                    nr = int(v)
                vids.append(int(nr))
        if len(vids) >= 3:
            triangles.append(vids[:3])

    triangles = np.array(triangles, dtype=int)

    if len(triangles) > 0 and triangles.max() >= len(pts):
        triangles -= 1
    elif len(triangles) > 0 and triangles.min() == 1:
        triangles -= 1

    return pts, triangles


def evaluate_gfu_on_mesh_vertices(mesh, gfu, pts):
    vals = np.full(len(pts), np.nan, dtype=float)
    for i, (x, y) in enumerate(pts):
        try:
            vals[i] = float(gfu(mesh(x, y)))
        except Exception:
            vals[i] = np.nan
    return vals


def save_fe_case(cfg, case_id, pts, triangles, T_fe, shape_code, p1, p2, cx, cy):
    os.makedirs(cfg.fe_outdir, exist_ok=True)
    fe_path = os.path.join(cfg.fe_outdir, f"fe_case_{case_id:06d}.npz")
    np.savez(
        fe_path,
        points=pts.astype(np.float32),
        triangles=triangles.astype(np.int32),
        temperature=T_fe.astype(np.float32),
        shape_code=np.array([shape_code], dtype=np.float32),
        p1=np.array([p1], dtype=np.float32),
        p2=np.array([p2], dtype=np.float32),
        cx=np.array([cx], dtype=np.float32),
        cy=np.array([cy], dtype=np.float32),
    )
    return fe_path


def save_geometry_fe_case(
    fe_dir,
    case,
    points,
    triangles,
    temperature,
    boundary_points,
    diagnostics,
):
    """Save one reusable FE case with structured geometry metadata."""
    os.makedirs(fe_dir, exist_ok=True)
    fe_path = os.path.join(fe_dir, f"fe_case_{case.case_id:06d}.npz")
    np.savez(
        fe_path,
        points=np.asarray(points, dtype=np.float32),
        triangles=np.asarray(triangles, dtype=np.int32),
        temperature=np.asarray(temperature, dtype=np.float32),
        boundary_points=np.asarray(boundary_points, dtype=np.float32),
        geometry_family=np.array([case.family], dtype=object),
        geometry_group=np.array([case.group], dtype=object),
        geometry_record_json=np.array(
            [json.dumps(case.to_record(), sort_keys=True)],
            dtype=object,
        ),
        area=np.array([diagnostics["area"]], dtype=np.float64),
        perimeter=np.array([diagnostics["perimeter"]], dtype=np.float64),
        domain_clearance=np.array(
            [diagnostics["domain_clearance"]],
            dtype=np.float64,
        ),
    )
    return fe_path
