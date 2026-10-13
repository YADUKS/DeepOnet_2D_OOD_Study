"""
Generate FE data using NGSolve and store a DeepONet-ready dataset.

This version supports:
    - circle datasets
    - rectangle datasets
    - combined datasets (circle + rectangle)
    - multiple dataset sizes
    - exact generation of requested case count
    - SDF, mask, temperature, FE storage
"""

import os
import numpy as np

from src.config import Settings
from src.fem.sampling import make_cartesian_grid, sample_case
from src.fem.fe_storage import save_fe_case
from src.fem.solver import export_vtk
from src.analysis.plots import plot_preview


# ============================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================

from src.data.design import latin_hypercube_sampling


def generate_lhs_circle_parameters(cfg, n_samples, seed_offset=0):
    lhs = latin_hypercube_sampling(n_samples, 3, seed=cfg.seed + seed_offset)
    params = []

    for zD, zx, zy in lhs:
        D = cfg.D_min + zD * (cfg.D_max - cfg.D_min)
        r = D / 2.0

        margin = 1.5 * cfg.maxh

        x_min = r + margin
        x_max = cfg.Lx - r - margin
        y_min = r + margin
        y_max = cfg.Ly - r - margin

        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)

        params.append((D, cx, cy))

    return params


def generate_lhs_rect_parameters(cfg, n_samples, seed_offset=0):
    lhs = latin_hypercube_sampling(n_samples, 4, seed=cfg.seed + 1000 + seed_offset)
    params = []

    for zW, zH, zx, zy in lhs:
        W = cfg.W_min + zW * (cfg.W_max - cfg.W_min)
        H = cfg.H_min + zH * (cfg.H_max - cfg.H_min)

        #margin = 1.5 * cfg.maxh
        margin = 0.003

        x_min = W / 2.0 + margin
        x_max = cfg.Lx - W / 2.0 - margin
        y_min = H / 2.0 + margin
        y_max = cfg.Ly - H / 2.0 - margin

        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)

        params.append((W, H, cx, cy))

    return params


# ============================================================
# PREVIEW
# ============================================================

def run_preview(cfg):
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.fe_outdir, exist_ok=True)

    if cfg.preview_shape == "circle":
        shape = "circle"
        p1, p2 = cfg.preview_D, 0.0
    elif cfg.preview_shape == "rect":
        shape = "rect"
        p1, p2 = cfg.preview_W, cfg.preview_H
    else:
        raise ValueError("cfg.preview_shape must be 'circle' or 'rect'.")

    mesh, gfu, sdf, mask, T, pts, triangles, T_fe = sample_case(
        cfg, shape, p1, p2, cfg.preview_cx, cfg.preview_cy
    )

    if cfg.export_vtk:
        export_vtk(cfg, mesh, gfu, "preview_case")

    plot_preview(
        cfg=cfg,
        shape=shape,
        p1=p1,
        p2=p2,
        cx=cfg.preview_cx,
        cy=cfg.preview_cy,
        sdf=sdf,
        mask=mask,
        T=T,
        pts=pts,
        triangles=triangles,
        T_fe=T_fe,
    )


# ============================================================
# BUILD CASE PLAN
# ============================================================

def build_case_plan(cfg):
    """
    Build the requested case list.

    Returns a list of tuples:
        (shape_name, p1, p2, cx, cy)

    where:
        circle -> p1 = D, p2 = 0
        rect   -> p1 = W, p2 = H
    """
    family = cfg.dataset_family.lower()
    requested = cfg.n_cases

    if family == "circle":
        candidate_count = max(int(np.ceil(1.5 * requested)), requested + 20)
        circle_params = generate_lhs_circle_parameters(cfg, candidate_count)

        case_plan = [("circle", D, 0.0, cx, cy) for (D, cx, cy) in circle_params]

    elif family == "rect":
        candidate_count = max(int(np.ceil(1.5 * requested)), requested + 20)
        rect_params = generate_lhs_rect_parameters(cfg, candidate_count)

        case_plan = [("rect", W, H, cx, cy) for (W, H, cx, cy) in rect_params]

    elif family == "combined":
        n_circle = requested // 2
        n_rect = requested - n_circle

        circle_candidate_count = max(int(np.ceil(1.5 * n_circle)), n_circle + 20)
        rect_candidate_count = max(int(np.ceil(1.5 * n_rect)), n_rect + 20)

        circle_params = generate_lhs_circle_parameters(cfg, circle_candidate_count)
        rect_params = generate_lhs_rect_parameters(cfg, rect_candidate_count)

        circle_cases = [("circle", D, 0.0, cx, cy) for (D, cx, cy) in circle_params[:circle_candidate_count]]
        rect_cases = [("rect", W, H, cx, cy) for (W, H, cx, cy) in rect_params[:rect_candidate_count]]

        case_plan = circle_cases + rect_cases

        rng = np.random.default_rng(cfg.seed + 999)
        rng.shuffle(case_plan)

    else:
        raise ValueError("cfg.dataset_family must be 'circle', 'rect', or 'combined'.")

    return case_plan


# ============================================================
# MAIN DATASET GENERATOR
# ============================================================

def generate_dataset(cfg):
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.fe_outdir, exist_ok=True)

    _, _, _, _, grid_points = make_cartesian_grid(cfg)

    sdf_all = []
    mask_all = []
    temp_all = []
    meta_all = []
    fe_paths = []
    valid_temperature_values = []

    case_plan = build_case_plan(cfg)
    requested_cases = cfg.n_cases
    accepted_cases = 0
    candidate_id = 0

    while accepted_cases < requested_cases:
        if candidate_id >= len(case_plan):
            raise RuntimeError(
                f"Could not generate enough valid cases for family='{cfg.dataset_family}'. "
                f"Accepted {accepted_cases} / {requested_cases}. "
                f"Increase oversampling or relax geometry margins."
            )

        shape, p1, p2, cx, cy = case_plan[candidate_id]
        candidate_id += 1

        print(
            f"[{accepted_cases + 1}/{requested_cases}] "
            f"Shape={shape}, p1={p1:.6f}, p2={p2:.6f}, cx={cx:.6f}, cy={cy:.6f}",
            flush=True
        )

        try:
            _, _, sdf, mask, T, pts, triangles, T_fe = sample_case(
                cfg, shape, p1, p2, cx, cy
            )
        except Exception as e:
            print("\nMeshing/solving failed for candidate:")
            print(f"shape = {shape}")
            print(f"p1    = {p1:.6f}")
            print(f"p2    = {p2:.6f}")
            print(f"cx    = {cx:.6f}")
            print(f"cy    = {cy:.6f}")
            print(f"Error = {e}")
            print("Skipping candidate and continuing...\n")
            continue

        shape_code = 0.0 if shape == "circle" else 1.0

        fe_path = save_fe_case(
            cfg=cfg,
            case_id=accepted_cases,
            pts=pts,
            triangles=triangles,
            T_fe=T_fe,
            shape_code=shape_code,
            p1=p1,
            p2=p2,
            cx=cx,
            cy=cy,
        )

        sdf_all.append((sdf / max(cfg.Lx, cfg.Ly)).ravel())
        mask_all.append(mask.ravel())
        temp_all.append(np.nan_to_num(T, nan=0.0).ravel())
        valid_temperature_values.append(T[mask > 0.5].ravel())
        meta_all.append([shape_code, p1, p2, cx, cy])
        fe_paths.append(fe_path)

        accepted_cases += 1

    sdf_all = np.array(sdf_all, dtype=np.float32)
    mask_all = np.array(mask_all, dtype=np.float32)
    temp_all = np.array(temp_all, dtype=np.float32)
    meta_all = np.array(meta_all, dtype=np.float32)
    fe_paths = np.array(fe_paths, dtype=object)

    valid_temperature_values = np.concatenate(valid_temperature_values)
    T_min = float(np.min(valid_temperature_values))
    T_max = float(np.max(valid_temperature_values))

    print(f"\nPhysical T_min from solid region = {T_min:.6f} K")
    print(f"Physical T_max from solid region = {T_max:.6f} K")

    temp_norm = (temp_all - T_min) / (T_max - T_min + 1e-12)
    temp_norm = temp_norm * mask_all

    np.savez(
        cfg.dataset_path,
        sdf=sdf_all,
        mask=mask_all,
        temp=temp_norm,
        meta=meta_all,
        fe_paths=fe_paths,
        coords=grid_points,
        T_min=np.array([T_min], dtype=np.float32),
        T_max=np.array([T_max], dtype=np.float32),
        Lx=np.array([cfg.Lx], dtype=np.float32),
        Ly=np.array([cfg.Ly], dtype=np.float32),
        Nx=np.array([cfg.Nx], dtype=np.int32),
        Ny=np.array([cfg.Ny], dtype=np.int32),
        maxh=np.array([cfg.maxh], dtype=np.float32),
        dataset_family=np.array([cfg.dataset_family], dtype=object),
        n_cases=np.array([accepted_cases], dtype=np.int32),
    )

    print(f"\nSaved dataset to: {cfg.dataset_path}")
    return cfg.dataset_path


# ============================================================
# GENERATE MULTIPLE DATASETS
# ============================================================

def generate_multiple_datasets(base_cfg, families, sizes, preview_first=False):
    from dataclasses import replace

    for family in families:
        for size in sizes:
            cfg = replace(base_cfg)
            cfg.dataset_family = family
            cfg.selected_dataset_size = size
            cfg.n_cases = size

            print("\n" + "=" * 80)
            print(f"Generating dataset | family = {family} | size = {size}")
            print("=" * 80)

            if preview_first:
                run_preview(cfg)

            generate_dataset(cfg)


if __name__ == "__main__":
    cfg = Settings()
    run_preview(cfg)
    ans = input("Continue to full dataset generation? [y/N]: ").strip().lower()
    if ans == "y":
        generate_dataset(cfg)
    else:
        print("Stopped after preview.")