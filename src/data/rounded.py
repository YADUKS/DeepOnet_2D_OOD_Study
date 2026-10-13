"""
Generate an external rounded-rectangle test dataset.

This dataset is for generalization testing only. It is saved separately from
the circle/rectangle training dataset and is not used during training.
"""

import os
import numpy as np

from src.config import Settings
from src.fem.sampling import make_cartesian_grid, sample_case


from src.data.design import latin_hypercube_sampling


def generate_lhs_rounded_rect_parameters(cfg, n_samples):
    lhs = latin_hypercube_sampling(
        n_samples=n_samples,
        n_dim=5,
        seed=cfg.rounded_test_seed,
    )
    params = []

    for zW, zH, zR, zx, zy in lhs:
        W = cfg.rounded_W_min + zW * (cfg.rounded_W_max - cfg.rounded_W_min)
        H = cfg.rounded_H_min + zH * (cfg.rounded_H_max - cfg.rounded_H_min)
        R_requested = cfg.rounded_R_min + zR * (cfg.rounded_R_max - cfg.rounded_R_min)
        R = min(R_requested, 0.45 * min(W, H))

        margin = cfg.rounded_margin
        x_min = W / 2.0 + margin
        x_max = cfg.Lx - W / 2.0 - margin
        y_min = H / 2.0 + margin
        y_max = cfg.Ly - H / 2.0 - margin

        if x_max <= x_min or y_max <= y_min:
            continue

        cx = x_min + zx * (x_max - x_min)
        cy = y_min + zy * (y_max - y_min)
        params.append((W, H, R, cx, cy))

    return params


def save_rounded_fe_case(cfg, case_id, pts, triangles, T_fe, W, H, R, cx, cy):
    os.makedirs(cfg.rounded_test_fe_outdir, exist_ok=True)
    fe_path = os.path.join(cfg.rounded_test_fe_outdir, f"rounded_rect_fe_case_{case_id:06d}.npz")
    np.savez(
        fe_path,
        points=pts.astype(np.float32),
        triangles=triangles.astype(np.int32),
        temperature=T_fe.astype(np.float32),
        shape_code=np.array([2.0], dtype=np.float32),
        p1=np.array([W], dtype=np.float32),
        p2=np.array([H], dtype=np.float32),
        p3=np.array([R], dtype=np.float32),
        cx=np.array([cx], dtype=np.float32),
        cy=np.array([cy], dtype=np.float32),
    )
    return fe_path


def generate_rounded_rect_test_dataset(cfg):
    os.makedirs(cfg.rounded_test_outdir, exist_ok=True)
    os.makedirs(cfg.rounded_test_fe_outdir, exist_ok=True)

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(
            f"Training dataset not found: {cfg.dataset_path}\n"
            "Generate the circle/rectangle training dataset first so its "
            "normalization scale can be reused."
        )

    training_data = np.load(cfg.dataset_path, allow_pickle=True)
    T_min = float(training_data["T_min"][0])
    T_max = float(training_data["T_max"][0])
    cfg.Lx = float(training_data["Lx"][0])
    cfg.Ly = float(training_data["Ly"][0])
    cfg.Nx = int(training_data["Nx"][0])
    cfg.Ny = int(training_data["Ny"][0])

    _, _, _, _, grid_points = make_cartesian_grid(cfg)

    requested_cases = int(cfg.rounded_test_n_cases)
    candidate_count = max(int(np.ceil(1.8 * requested_cases)), requested_cases + 20)
    case_plan = generate_lhs_rounded_rect_parameters(cfg, candidate_count)

    sdf_all = []
    mask_all = []
    temp_all = []
    meta_all = []
    fe_paths = []
    valid_temperature_values = []

    accepted_cases = 0
    candidate_id = 0

    while accepted_cases < requested_cases:
        if candidate_id >= len(case_plan):
            raise RuntimeError(
                f"Could not generate enough rounded-rectangle cases. "
                f"Accepted {accepted_cases} / {requested_cases}. "
                "Increase oversampling or relax rounded rectangle margins."
            )

        W, H, R, cx, cy = case_plan[candidate_id]
        candidate_id += 1

        print(
            f"[{accepted_cases + 1}/{requested_cases}] "
            f"Shape=rounded_rect, W={W:.6f}, H={H:.6f}, R={R:.6f}, "
            f"cx={cx:.6f}, cy={cy:.6f}",
            flush=True,
        )

        try:
            _, _, sdf, mask, T, pts, triangles, T_fe = sample_case(
                cfg,
                "rounded_rect",
                W,
                H,
                cx,
                cy,
                p3=R,
            )
        except Exception as exc:
            print("\nMeshing/solving failed for rounded rectangle candidate:")
            print(f"W     = {W:.6f}")
            print(f"H     = {H:.6f}")
            print(f"R     = {R:.6f}")
            print(f"cx    = {cx:.6f}")
            print(f"cy    = {cy:.6f}")
            print(f"Error = {exc}")
            print("Skipping candidate and continuing...\n")
            continue

        fe_path = save_rounded_fe_case(
            cfg=cfg,
            case_id=accepted_cases,
            pts=pts,
            triangles=triangles,
            T_fe=T_fe,
            W=W,
            H=H,
            R=R,
            cx=cx,
            cy=cy,
        )

        sdf_all.append((sdf / max(cfg.Lx, cfg.Ly)).ravel())
        mask_all.append(mask.ravel())
        temp_all.append(np.nan_to_num(T, nan=0.0).ravel())
        valid_temperature_values.append(T[mask > 0.5].ravel())
        meta_all.append([2.0, W, H, R, cx, cy])
        fe_paths.append(fe_path)

        accepted_cases += 1

    sdf_all = np.asarray(sdf_all, dtype=np.float32)
    mask_all = np.asarray(mask_all, dtype=np.float32)
    temp_all = np.asarray(temp_all, dtype=np.float32)
    meta_all = np.asarray(meta_all, dtype=np.float32)
    fe_paths = np.asarray(fe_paths, dtype=object)

    rounded_temperature_values = np.concatenate(valid_temperature_values)
    rounded_T_min = float(np.min(rounded_temperature_values))
    rounded_T_max = float(np.max(rounded_temperature_values))

    temp_norm = (temp_all - T_min) / (T_max - T_min + 1e-12)
    temp_norm = temp_norm * mask_all

    np.savez(
        cfg.rounded_test_dataset_path,
        sdf=sdf_all,
        mask=mask_all,
        temp=temp_norm,
        meta=meta_all,
        fe_paths=fe_paths,
        coords=grid_points,
        T_min=np.array([T_min], dtype=np.float32),
        T_max=np.array([T_max], dtype=np.float32),
        rounded_T_min=np.array([rounded_T_min], dtype=np.float32),
        rounded_T_max=np.array([rounded_T_max], dtype=np.float32),
        Lx=np.array([cfg.Lx], dtype=np.float32),
        Ly=np.array([cfg.Ly], dtype=np.float32),
        Nx=np.array([cfg.Nx], dtype=np.int32),
        Ny=np.array([cfg.Ny], dtype=np.int32),
        maxh=np.array([cfg.maxh], dtype=np.float32),
        dataset_family=np.array(["rounded_rect_external_test"], dtype=object),
        n_cases=np.array([accepted_cases], dtype=np.int32),
        source_training_dataset=np.array([cfg.dataset_path], dtype=object),
    )

    print(f"\nTraining normalization T_min = {T_min:.6f} K")
    print(f"Training normalization T_max = {T_max:.6f} K")
    print(f"Rounded test actual T_min    = {rounded_T_min:.6f} K")
    print(f"Rounded test actual T_max    = {rounded_T_max:.6f} K")
    print(f"Saved rounded-rectangle test dataset to: {cfg.rounded_test_dataset_path}")
    return cfg.rounded_test_dataset_path


if __name__ == "__main__":
    generate_rounded_rect_test_dataset(Settings())
