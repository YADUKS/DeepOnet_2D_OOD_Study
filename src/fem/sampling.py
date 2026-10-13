"""
Grid construction and sampling FE solutions onto a fixed Cartesian grid.
"""

import numpy as np



def make_cartesian_grid(cfg):
    xgrid = np.linspace(0.0, cfg.Lx, cfg.Nx)
    ygrid = np.linspace(0.0, cfg.Ly, cfg.Ny)
    Xg, Yg = np.meshgrid(xgrid, ygrid, indexing="xy")
    grid_points = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
    return xgrid, ygrid, Xg, Yg, grid_points


def sample_case(cfg, shape, p1, p2, cx, cy, p3=0.0):
    from src.fem.geometry import build_mesh_and_sdf
    from src.fem.solver import solve_temperature
    from src.fem.fe_storage import get_mesh_points_and_triangles, evaluate_gfu_on_mesh_vertices
    _, _, Xg, Yg, _ = make_cartesian_grid(cfg)

    mesh, sdf = build_mesh_and_sdf(cfg, shape, p1, p2, cx, cy, Xg, Yg, p3=p3)
    gfu = solve_temperature(cfg, mesh)

    mask = (sdf >= 0.0).astype(np.float32)
    T = np.full_like(sdf, np.nan, dtype=np.float32)

    for j in range(cfg.Ny):
        for i in range(cfg.Nx):
            if mask[j, i] > 0.5:
                x = Xg[j, i]
                y = Yg[j, i]
                try:
                    T[j, i] = float(gfu(mesh(x, y)))
                except Exception:
                    T[j, i] = np.nan

    pts, triangles = get_mesh_points_and_triangles(mesh)
    T_fe = evaluate_gfu_on_mesh_vertices(mesh, gfu, pts)

    return mesh, gfu, sdf, mask, T, pts, triangles, T_fe


def _sample_structured_geometry_case(cfg, case, mesh_sdf_builder):
    from src.fem.solver import solve_temperature
    from src.fem.fe_storage import get_mesh_points_and_triangles, evaluate_gfu_on_mesh_vertices
    _, _, Xg, Yg, _ = make_cartesian_grid(cfg)
    mesh, sdf, boundary_points, diagnostics = mesh_sdf_builder(
        cfg,
        case,
        Xg,
        Yg,
    )
    gfu = solve_temperature(cfg, mesh)

    mask = (sdf >= 0.0).astype(np.float32)
    temperature = np.full_like(sdf, np.nan, dtype=np.float32)

    for j in range(cfg.Ny):
        for i in range(cfg.Nx):
            if mask[j, i] > 0.5:
                try:
                    temperature[j, i] = float(gfu(mesh(Xg[j, i], Yg[j, i])))
                except Exception:
                    temperature[j, i] = np.nan

    points, triangles = get_mesh_points_and_triangles(mesh)
    temperature_fe = evaluate_gfu_on_mesh_vertices(mesh, gfu, points)
    return (
        mesh,
        gfu,
        sdf,
        mask,
        temperature,
        points,
        triangles,
        temperature_fe,
        boundary_points,
        diagnostics,
    )


def sample_geometry_case(cfg, case):
    """Solve and sample a structured simple-training geometry case."""
    from src.fem.geometry import build_simple_mesh_and_sdf
    return _sample_structured_geometry_case(
        cfg,
        case,
        build_simple_mesh_and_sdf,
    )


def sample_ood_geometry_case(cfg, case):
    """Solve and sample a structured OOD geometry case."""
    from src.fem.geometry import build_ood_mesh_and_sdf
    return _sample_structured_geometry_case(
        cfg,
        case,
        build_ood_mesh_and_sdf,
    )


def bilinear_interpolate_regular_grid(field, xgrid, ygrid, xq, yq):
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    xq = np.clip(xq, xgrid[0], xgrid[-1])
    yq = np.clip(yq, ygrid[0], ygrid[-1])

    ix = np.searchsorted(xgrid, xq, side="right") - 1
    iy = np.searchsorted(ygrid, yq, side="right") - 1

    ix = np.clip(ix, 0, len(xgrid) - 2)
    iy = np.clip(iy, 0, len(ygrid) - 2)

    x0 = xgrid[ix]
    x1 = xgrid[ix + 1]
    y0 = ygrid[iy]
    y1 = ygrid[iy + 1]

    tx = (xq - x0) / (x1 - x0 + 1e-15)
    ty = (yq - y0) / (y1 - y0 + 1e-15)

    f00 = field[iy, ix]
    f10 = field[iy, ix + 1]
    f01 = field[iy + 1, ix]
    f11 = field[iy + 1, ix + 1]

    return ((1 - tx) * (1 - ty) * f00 +
            tx * (1 - ty) * f10 +
            (1 - tx) * ty * f01 +
            tx * ty * f11)
