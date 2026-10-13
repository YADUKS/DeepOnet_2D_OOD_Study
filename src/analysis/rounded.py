"""
Evaluate the trained SDF-in-trunk DeepONet on unseen rounded-rectangle holes.

The model is trained on the usual circle/rectangle dataset. This script loads
the separate rounded-rectangle external test dataset and measures
generalization to a geometry family that was not used for training.
"""

import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib.tri as mtri

try:
    import psutil
except ImportError:
    psutil = None

from src.config import Settings
from src.models.deeponet import SimpleDeepONet, masked_mse
from src.fem.sampling import make_cartesian_grid, bilinear_interpolate_regular_grid
from src.analysis.plots import (
    get_test_case_output_dir,
    plot_test_case_combined,
    save_test_case_publishable_figures,
)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    return device


def normalize_trunk_coordinates(coords, Lx, Ly):
    coords_norm = coords.copy()
    coords_norm[:, 0] = coords_norm[:, 0] / Lx
    coords_norm[:, 1] = coords_norm[:, 1] / Ly
    return coords_norm.astype(np.float32)


def build_case_dependent_trunk_input(coords_t, sdf_b):
    batch_size = sdf_b.shape[0]
    coords_batch = coords_t.unsqueeze(0).expand(batch_size, -1, -1)
    sdf_feature = sdf_b.unsqueeze(-1)
    return torch.cat([coords_batch, sdf_feature], dim=-1)


def get_cpu_memory_mb():
    if psutil is None:
        return np.nan
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return np.nan


def compute_metrics(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    valid = np.isfinite(true_values) & np.isfinite(pred_values)

    true_valid = true_values[valid]
    pred_valid = pred_values[valid]
    diff = pred_valid - true_valid

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(100.0 * np.abs(diff) / (np.abs(true_valid) + 1e-12)))
    rmse = float(np.sqrt(mse))
    l2_error = float(np.linalg.norm(diff))
    relative_l2 = float(l2_error / (np.linalg.norm(true_valid) + 1e-12))
    max_abs_error = float(np.max(np.abs(diff)))
    max_relative_error = float(np.max(100.0 * np.abs(diff) / (np.abs(true_valid) + 1e-12)))

    return {
        "mse": mse,
        "mae": mae,
        "mape_percent": mape,
        "rmse": rmse,
        "l2_error": l2_error,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
        "max_relative_error_percent": max_relative_error,
        "n_valid_points": int(valid.sum()),
    }


def save_metrics_csv(cfg, rows):
    fieldnames = [
        "dataset_family",
        "dataset_size",
        "rounded_test_case_number",
        "case_id",
        "shape",
        "W",
        "H",
        "R",
        "cx",
        "cy",
        "normalized_masked_mse",
        "mse",
        "mae",
        "mape_percent",
        "rmse",
        "l2_error",
        "relative_l2",
        "max_abs_error",
        "max_relative_error_percent",
        "n_valid_points",
    ]

    with open(cfg.rounded_test_metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {cfg.rounded_test_metrics_csv_path}")


def save_summary_txt(cfg, rows, normalized_test_loss, cpu_memory_mb, gpu_peak_memory_mb):
    rmse_values = np.asarray([row["rmse"] for row in rows], dtype=np.float64)
    mape_values = np.asarray([row["mape_percent"] for row in rows], dtype=np.float64)
    mae_values = np.asarray([row["mae"] for row in rows], dtype=np.float64)
    rel_l2_values = np.asarray([row["relative_l2"] for row in rows], dtype=np.float64)

    with open(cfg.rounded_test_summary_path, "w") as f:
        f.write("Rounded-Rectangle External Test Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training dataset family : {cfg.dataset_family}\n")
        f.write(f"Training dataset size   : {cfg.selected_dataset_size}\n")
        f.write(f"Model path              : {cfg.model_path}\n")
        f.write(f"External dataset path   : {cfg.rounded_test_dataset_path}\n")
        f.write(f"Number of test cases    : {len(rows)}\n")
        f.write(f"Selected case number    : {cfg.rounded_test_case_number}\n")
        f.write(f"Normalized masked MSE   : {normalized_test_loss:.8e}\n")
        f.write("\n")
        f.write("Mean metrics on actual FE mesh\n")
        f.write("-" * 60 + "\n")
        f.write(f"MAE [K]                 : {np.mean(mae_values):.8e}\n")
        f.write(f"RMSE [K]                : {np.mean(rmse_values):.8e}\n")
        f.write(f"MAPE [%]                : {np.mean(mape_values):.8e}\n")
        f.write(f"Relative L2             : {np.mean(rel_l2_values):.8e}\n")
        f.write("\n")
        f.write("Worst-case metrics on actual FE mesh\n")
        f.write("-" * 60 + "\n")
        worst_rmse_idx = int(np.argmax(rmse_values))
        worst_mape_idx = int(np.argmax(mape_values))
        f.write(f"Worst RMSE case         : {rows[worst_rmse_idx]['case_id']}\n")
        f.write(f"Worst RMSE [K]          : {rmse_values[worst_rmse_idx]:.8e}\n")
        f.write(f"Worst MAPE case         : {rows[worst_mape_idx]['case_id']}\n")
        f.write(f"Worst MAPE [%]          : {mape_values[worst_mape_idx]:.8e}\n")
        f.write("\n")
        f.write("Memory\n")
        f.write("-" * 60 + "\n")
        f.write(f"CPU memory [MB]         : {cpu_memory_mb:.4f}\n")
        f.write(f"GPU peak memory [MB]    : {gpu_peak_memory_mb:.4f}\n")

    print(f"Saved: {cfg.rounded_test_summary_path}")


def write_selected_case_summary_txt(
    cfg,
    case_id,
    W,
    H,
    R,
    cx,
    cy,
    normalized_test_loss,
    selected_metrics,
    cpu_memory_mb,
    gpu_peak_memory_mb,
):
    case_dir = get_test_case_output_dir(cfg, case_id)
    summary_path = os.path.join(
        case_dir,
        f"test_case_number_{int(cfg.test_case_number):04d}_case_id_{int(case_id):06d}_summary.txt",
    )

    memory_used_mb = gpu_peak_memory_mb if torch.cuda.is_available() else cpu_memory_mb

    with open(summary_path, "w") as f:
        f.write("DeepONet Rounded-Rectangle External Test Case Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training dataset family : {cfg.dataset_family}\n")
        f.write(f"Training dataset size   : {cfg.selected_dataset_size}\n")
        f.write(f"Training dataset path   : {cfg.dataset_path}\n")
        f.write(f"External dataset path   : {cfg.rounded_test_dataset_path}\n")
        f.write(f"Model path              : {cfg.model_path}\n\n")

        f.write("Selected external test case\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test case number        : {int(cfg.test_case_number)}\n")
        f.write(f"Actual case ID          : {int(case_id)}\n")
        f.write("Shape                   : rounded_rect\n")
        f.write(f"Width W                 : {W:.8f} m\n")
        f.write(f"Height H                : {H:.8f} m\n")
        f.write(f"Corner radius R         : {R:.8f} m\n")
        f.write(f"Centre cx               : {cx:.8f} m\n")
        f.write(f"Centre cy               : {cy:.8f} m\n\n")

        f.write("Model input/output description\n")
        f.write("-" * 70 + "\n")
        f.write("Branch input            : SDF only\n")
        f.write("Trunk input             : x, y, SDF(x,y)\n")
        f.write("Output                  : Normalized temperature field\n")
        f.write("Evaluation field        : Prediction interpolated onto actual FE mesh\n\n")

        f.write("Error metrics on FE mesh\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean external test loss : {normalized_test_loss:.8e}\n")
        f.write(f"Case normalized MSE     : {selected_metrics['normalized_masked_mse']:.8e}\n")
        f.write(f"MSE                     : {selected_metrics['mse']:.8e}\n")
        f.write(f"MAE                     : {selected_metrics['mae']:.8e} K\n")
        f.write(f"MAPE                    : {selected_metrics['mape_percent']:.8e} %\n")
        f.write(f"RMSE                    : {selected_metrics['rmse']:.8e} K\n")
        f.write(f"L2 error                : {selected_metrics['l2_error']:.8e}\n")
        f.write(f"Relative L2 error       : {selected_metrics['relative_l2']:.8e}\n")
        f.write(f"Max absolute error      : {selected_metrics['max_abs_error']:.8e} K\n")
        f.write(f"Max relative error      : {selected_metrics['max_relative_error_percent']:.8e} %\n\n")

        f.write("Memory information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Memory used             : {memory_used_mb:.4f} MB\n")
        f.write(f"CPU memory used         : {cpu_memory_mb:.4f} MB\n")
        f.write(f"GPU peak memory used    : {gpu_peak_memory_mb:.4f} MB\n\n")

        f.write("Saved figure folder\n")
        f.write("-" * 70 + "\n")
        f.write(f"{case_dir}\n")

    print(f"Saved selected rounded test-case summary text file to: {summary_path}")
    return summary_path


def evaluate_rounded_rect_test_cases(cfg):
    os.makedirs(cfg.rounded_test_outdir, exist_ok=True)

    if not os.path.exists(cfg.rounded_test_dataset_path):
        raise FileNotFoundError(
            f"Rounded-rectangle test dataset not found: {cfg.rounded_test_dataset_path}\n"
            "Run generate_rounded_rect_test_dataset.py first."
        )

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(
            f"Trained model not found: {cfg.model_path}\n"
            "Train the SDF-in-trunk DeepONet model first."
        )

    device = get_device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data = np.load(cfg.rounded_test_dataset_path, allow_pickle=True)
    sdf_all = data["sdf"].astype(np.float32)
    mask_all = data["mask"].astype(np.float32)
    temp_all = data["temp"].astype(np.float32)
    meta_all = data["meta"].astype(np.float32)
    fe_paths = data["fe_paths"]
    coords = data["coords"].astype(np.float32)

    T_min = float(data["T_min"][0])
    T_max = float(data["T_max"][0])

    cfg.Lx = float(data["Lx"][0])
    cfg.Ly = float(data["Ly"][0])
    cfg.Nx = int(data["Nx"][0])
    cfg.Ny = int(data["Ny"][0])

    coords = normalize_trunk_coordinates(coords, cfg.Lx, cfg.Ly)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    xgrid, ygrid, _, _, _ = make_cartesian_grid(cfg)

    model = SimpleDeepONet(
        branch_in=sdf_all.shape[1],
        trunk_in=3,
        latent=cfg.latent_dim,
    ).to(device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

    if cfg.rounded_test_case_number >= sdf_all.shape[0]:
        raise ValueError(
            f"cfg.rounded_test_case_number = {cfg.rounded_test_case_number} is too large. "
            f"Available rounded test cases = {sdf_all.shape[0]}"
        )

    rows = []
    normalized_losses = []
    selected_payload = None

    with torch.no_grad():
        for case_id in range(sdf_all.shape[0]):
            sdf_case = torch.tensor(sdf_all[case_id:case_id + 1], dtype=torch.float32, device=device)
            mask_case = torch.tensor(mask_all[case_id:case_id + 1], dtype=torch.float32, device=device)
            temp_case = torch.tensor(temp_all[case_id:case_id + 1], dtype=torch.float32, device=device)

            trunk_case = build_case_dependent_trunk_input(coords_t, sdf_case)
            pred_norm_t = model(sdf_case, trunk_case)
            loss = masked_mse(pred_norm_t, temp_case, mask_case)
            normalized_losses.append(float(loss.item()))

            pred_norm = pred_norm_t.cpu().numpy().reshape(cfg.Ny, cfg.Nx)
            true_norm = temp_all[case_id].reshape(cfg.Ny, cfg.Nx)
            mask = mask_all[case_id].reshape(cfg.Ny, cfg.Nx)

            pred_grid = pred_norm * (T_max - T_min) + T_min
            true_grid = true_norm * (T_max - T_min) + T_min

            _, W, H, R, cx, cy = meta_all[case_id]
            fe_data = np.load(str(fe_paths[case_id]))
            pts = fe_data["points"].astype(np.float64)
            triangles = fe_data["triangles"].astype(np.int32)
            true_fe = fe_data["temperature"].astype(np.float64)

            pred_fe = bilinear_interpolate_regular_grid(
                pred_grid,
                xgrid,
                ygrid,
                pts[:, 0],
                pts[:, 1],
            )

            metrics = compute_metrics(true_fe, pred_fe)
            row = {
                    "dataset_family": "rounded_rect_external_test",
                    "dataset_size": int(sdf_all.shape[0]),
                    "rounded_test_case_number": int(case_id),
                    "case_id": int(case_id),
                    "shape": "rounded_rect",
                    "W": float(W),
                    "H": float(H),
                    "R": float(R),
                    "cx": float(cx),
                    "cy": float(cy),
                    "normalized_masked_mse": float(loss.item()),
                    **metrics,
            }
            rows.append(row)

            if case_id == int(cfg.rounded_test_case_number):
                signed_fe = pred_fe - true_fe
                abs_fe = np.abs(signed_fe)
                relative_fe = 100.0 * abs_fe / (np.abs(true_fe) + 1e-12)

                tri = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)
                tri_mask = np.any(np.isnan(true_fe[triangles]), axis=1)
                tri.set_mask(tri_mask)

                true_masked = true_grid.copy()
                pred_masked = pred_grid.copy()
                true_masked[mask < 0.5] = np.nan
                pred_masked[mask < 0.5] = np.nan

                selected_payload = {
                    "case_id": case_id,
                    "W": float(W),
                    "H": float(H),
                    "R": float(R),
                    "cx": float(cx),
                    "cy": float(cy),
                    "tri": tri,
                    "true_fe": true_fe,
                    "pred_fe": pred_fe,
                    "true_masked": true_masked,
                    "pred_masked": pred_masked,
                    "signed_fe": signed_fe,
                    "abs_fe": abs_fe,
                    "relative_fe": relative_fe,
                    "vmin": float(np.nanmin(true_fe)),
                    "vmax": float(np.nanmax(true_fe)),
                    "metrics": row,
                }

    normalized_test_loss = float(np.mean(normalized_losses))
    save_metrics_csv(cfg, rows)

    cfg.active_test_outdir = cfg.rounded_test_outdir
    cfg.test_case_number = int(cfg.rounded_test_case_number)

    plot_test_case_combined(
        cfg=cfg,
        case_id=selected_payload["case_id"],
        shape_name="rounded_rect",
        p1=selected_payload["W"],
        p2=selected_payload["H"],
        p3=selected_payload["R"],
        cx=selected_payload["cx"],
        cy=selected_payload["cy"],
        tri=selected_payload["tri"],
        true_fe=selected_payload["true_fe"],
        pred_fe=selected_payload["pred_fe"],
        true_masked=selected_payload["true_masked"],
        pred_masked=selected_payload["pred_masked"],
        signed_fe=selected_payload["signed_fe"],
        abs_fe=selected_payload["abs_fe"],
        relative_fe=selected_payload["relative_fe"],
        vmin=selected_payload["vmin"],
        vmax=selected_payload["vmax"],
    )

    save_test_case_publishable_figures(
        cfg=cfg,
        case_id=selected_payload["case_id"],
        shape_name="rounded_rect",
        p1=selected_payload["W"],
        p2=selected_payload["H"],
        p3=selected_payload["R"],
        cx=selected_payload["cx"],
        cy=selected_payload["cy"],
        tri=selected_payload["tri"],
        true_fe=selected_payload["true_fe"],
        pred_fe=selected_payload["pred_fe"],
        signed_fe=selected_payload["signed_fe"],
        abs_fe=selected_payload["abs_fe"],
        relative_fe=selected_payload["relative_fe"],
        vmin=selected_payload["vmin"],
        vmax=selected_payload["vmax"],
    )

    cpu_memory_mb = get_cpu_memory_mb()
    gpu_peak_memory_mb = get_gpu_peak_memory_mb()
    save_summary_txt(cfg, rows, normalized_test_loss, cpu_memory_mb, gpu_peak_memory_mb)
    write_selected_case_summary_txt(
        cfg=cfg,
        case_id=selected_payload["case_id"],
        W=selected_payload["W"],
        H=selected_payload["H"],
        R=selected_payload["R"],
        cx=selected_payload["cx"],
        cy=selected_payload["cy"],
        normalized_test_loss=normalized_test_loss,
        selected_metrics=selected_payload["metrics"],
        cpu_memory_mb=cpu_memory_mb,
        gpu_peak_memory_mb=gpu_peak_memory_mb,
    )

    print("\nRounded-rectangle external test summary")
    print("---------------------------------------")
    print(f"Cases evaluated       : {len(rows)}")
    print(f"Selected case         : {cfg.rounded_test_case_number}")
    print(f"Normalized masked MSE : {normalized_test_loss:.6e}")
    print(f"Mean RMSE [K]         : {np.mean([r['rmse'] for r in rows]):.6e}")
    print(f"Mean MAPE [%]         : {np.mean([r['mape_percent'] for r in rows]):.6e}")

    return rows


if __name__ == "__main__":
    evaluate_rounded_rect_test_cases(Settings())
