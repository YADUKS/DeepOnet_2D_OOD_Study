"""
Evaluate all test cases using the saved best DeepONet model.

This script:
    1. Loads dataset.npz from cfg.dataset_path
    2. Loads best_deeponet.pt from cfg.model_path
    3. Uses the same train/val/test split as training
    4. Evaluates every unseen test case
    5. Computes:
        - MSE
        - MAE
        - MAPE
        - RMSE
        - L2 error
        - Relative L2 error
        - Maximum absolute error
        - Maximum relative error
    6. Saves:
        - all_test_case_metrics.csv
        - all_test_case_metrics_summary.txt

Important:
    This file does NOT import sampling.py, geometry.py, solver.py,
    NGSolve, or Netgen.

    It only needs:
        - numpy
        - torch
        - psutil optional
        - dataset.npz
        - best_deeponet.pt
        - saved fe_cases/*.npz
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import time
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt

#from deeponet_ngsolve_modular import model
try:
    import psutil
except ImportError:
    psutil = None

from src.config import Settings
from src.data.dataset import make_loaders
from src.models.deeponet import SimpleDeepONet, masked_mse


# ============================================================
# DEVICE AND MEMORY HELPERS
# ============================================================

def get_device():
    """
    Return CUDA if available, otherwise CPU.

    No printing is done (user output supressed, silent run).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_trunk_coordinates(coords, Lx, Ly):
    coords_norm = coords.copy()

    coords_norm[:, 0] = coords_norm[:, 0] / Lx
    coords_norm[:, 1] = coords_norm[:, 1] / Ly

    return coords_norm.astype(np.float32)

def build_case_dependent_trunk_input(coords_t, sdf_b):
    batch_size = sdf_b.shape[0]
    coords_batch = coords_t.unsqueeze(0).expand(batch_size, -1, -1)
    sdf_feature = sdf_b.unsqueeze(-1)
    trunk_b = torch.cat([coords_batch, sdf_feature], dim=-1)
    return trunk_b

def get_cpu_memory_mb():
    """
    Return current process CPU memory use in MB.
    """
    if psutil is None:
        return np.nan

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_peak_memory_mb():
    """
    Return peak GPU memory allocated by PyTorch in MB.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    return np.nan


# ============================================================
# GRID AND INTERPOLATION HELPERS
# ============================================================

def make_cartesian_grid_from_dataset(Lx, Ly, Nx, Ny):
    """
    Create the same Cartesian grid used during dataset generation.

    This avoids importing sampling.py, because sampling.py imports
    geometry.py and solver.py, which then load NGSolve/Netgen.
    """
    xgrid = np.linspace(0.0, Lx, Nx)
    ygrid = np.linspace(0.0, Ly, Ny)

    return xgrid, ygrid


def bilinear_interpolate_regular_grid(field, xgrid, ygrid, xq, yq):
    """
    Bilinear interpolation from a regular Cartesian grid to query points.

    Parameters
    ----------
    field : ndarray, shape (Ny, Nx)
        Field values on regular grid.

    xgrid : ndarray, shape (Nx,)
        x coordinates of grid.

    ygrid : ndarray, shape (Ny,)
        y coordinates of grid.

    xq, yq : ndarray
        Query coordinates.

    Returns
    -------
    interpolated : ndarray
        Interpolated values at query coordinates.
    """

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

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.std(values))


def safe_max(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.max(values))


def compute_metrics(true_values, pred_values):
    """
    Compute error metrics between true and predicted values.

    Metrics are computed only where both true and predicted values are finite.
    """

    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)

    valid = np.isfinite(true_values) & np.isfinite(pred_values)

    true_valid = true_values[valid]
    pred_valid = pred_values[valid]

    if len(true_valid) == 0:
        return {
            "mse": np.nan,
            "mae": np.nan,
            "mape_percent": np.nan,
            "rmse": np.nan,
            "l2_error": np.nan,
            "relative_l2": np.nan,
            "max_abs_error": np.nan,
            "max_relative_error_percent": np.nan,
            "num_valid_points": 0,
        }

    diff = pred_valid - true_valid
    abs_diff = np.abs(diff)

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))
    mape = float(np.mean(100.0 * abs_diff / (np.abs(true_valid) + 1e-12)))
    rmse = float(np.sqrt(mse))
    l2_error = float(np.linalg.norm(diff))
    relative_l2 = float(l2_error / (np.linalg.norm(true_valid) + 1e-12))
    max_abs_error = float(np.max(abs_diff))
    max_relative_error = float(
        np.max(100.0 * abs_diff / (np.abs(true_valid) + 1e-12))
    )

    return {
        "mse": mse,
        "mae": mae,
        "mape_percent": mape,
        "rmse": rmse,
        "l2_error": l2_error,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
        "max_relative_error_percent": max_relative_error,
        "num_valid_points": int(len(true_valid)),
    }


# ============================================================
# FILE WRITING HELPERS
# ============================================================

def save_metrics_csv(csv_path, rows):
    """
    Save per-test-case metrics to CSV.
    """

    fieldnames = [
        "dataset_family",
        "dataset_size",
        "test_case_number",
        "case_id",
        "shape",
        "p1",
        "p2",
        "cx",
        "cy",
        "mse",
        "mae",
        "mape_percent",
        "rmse",
        "l2_error",
        "relative_l2",
        "max_abs_error",
        "max_relative_error_percent",
        "num_valid_points",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def save_rmse_mape_plot_and_csv(cfg, rows):
    """
    Save a separate CSV and figure showing RMSE and MAPE for all test cases.

    Outputs:
        all_test_case_rmse_mape_values.csv
        all_test_case_rmse_mape_plot.png
        all_test_case_rmse_mape_plot.pdf
    """

    csv_path = os.path.join(cfg.outdir, "all_test_case_rmse_mape_values.csv")
    png_path = os.path.join(cfg.outdir, "all_test_case_rmse_mape_plot.png")
    pdf_path = os.path.join(cfg.outdir, "all_test_case_rmse_mape_plot.pdf")

    test_case_numbers = np.array([r["test_case_number"] for r in rows], dtype=int)
    case_ids = np.array([r["case_id"] for r in rows], dtype=int)
    rmse_values = np.array([r["rmse"] for r in rows], dtype=float)
    mape_values = np.array([r["mape_percent"] for r in rows], dtype=float)
    mae_values = np.array([r["mae"] for r in rows], dtype=float)
    shape_values = [r["shape"] for r in rows]

    # ------------------------------------------------------------
    # Save compact CSV for plotting/comparison
    # ------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "test_case_number",
            "case_id",
            "shape",
            "rmse_K",
            "mape_percent",
            "mae_K",
        ])

        for i in range(len(rows)):
            writer.writerow([
                int(test_case_numbers[i]),
                int(case_ids[i]),
                shape_values[i],
                float(rmse_values[i]),
                float(mape_values[i]),
                float(mae_values[i]),
            ])

    # ------------------------------------------------------------
    # Find worst cases for annotation
    # ------------------------------------------------------------
    worst_rmse_idx = int(np.nanargmax(rmse_values))
    worst_mape_idx = int(np.nanargmax(mape_values))

    # ------------------------------------------------------------
    # Plot RMSE and MAPE in one figure using two y-axes
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        test_case_numbers,
        rmse_values,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="RMSE [K]",
    )

    ax1.set_xlabel("Test case number")
    ax1.set_ylabel("RMSE [K]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        test_case_numbers,
        mape_values,
        marker="s",
        markersize=3,
        linewidth=1.5,
        linestyle="--",
        label="MAPE [%]",
    )

    ax2.set_ylabel("MAPE [%]")

    # ------------------------------------------------------------
    # Mark worst RMSE and MAPE cases
    # ------------------------------------------------------------
    ax1.scatter(
        test_case_numbers[worst_rmse_idx],
        rmse_values[worst_rmse_idx],
        s=80,
        color="red",
        zorder=5,
    )

    ax2.scatter(
        test_case_numbers[worst_mape_idx],
        mape_values[worst_mape_idx],
        s=80,
        color="black",
        zorder=5,
    )

    # ax1.annotate(
    #     f"Worst RMSE\ncase_id={case_ids[worst_rmse_idx]}\nRMSE={rmse_values[worst_rmse_idx]:.2f} K",
    #     xy=(test_case_numbers[worst_rmse_idx], rmse_values[worst_rmse_idx]),
    #     xytext=(10, 20),
    #     textcoords="offset points",
    #     fontsize=9,
    #     arrowprops=dict(arrowstyle="->", lw=1.0),
    # )

    # ax2.annotate(
    #     f"Worst MAPE\ncase_id={case_ids[worst_mape_idx]}\nMAPE={mape_values[worst_mape_idx]:.2f} %",
    #     xy=(test_case_numbers[worst_mape_idx], mape_values[worst_mape_idx]),
    #     xytext=(10, -45),
    #     textcoords="offset points",
    #     fontsize=9,
    #     arrowprops=dict(arrowstyle="->", lw=1.0),
    # )

    # ------------------------------------------------------------
    # Combined legend
    # ------------------------------------------------------------
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )

    fig.suptitle(
        f"RMSE and MAPE over all test cases | {cfg.dataset_family}_{cfg.selected_dataset_size}",
        fontsize=13,
    )

    plt.tight_layout()

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def save_summary_txt(
    txt_path,
    cfg,
    rows,
    test_loss,
    elapsed_time_sec,
    cpu_memory_mb,
    gpu_peak_memory_mb,
    csv_path,
):
    """
    Save summary of all test-case metrics to a text file.
    """

    mse_values = [r["mse"] for r in rows]
    mae_values = [r["mae"] for r in rows]
    mape_values = [r["mape_percent"] for r in rows]
    rmse_values = [r["rmse"] for r in rows]
    l2_values = [r["l2_error"] for r in rows]
    rel_l2_values = [r["relative_l2"] for r in rows]
    max_abs_values = [r["max_abs_error"] for r in rows]
    max_rel_values = [r["max_relative_error_percent"] for r in rows]

    circle_count = sum(1 for r in rows if r["shape"] == "circle")
    rect_count = sum(1 for r in rows if r["shape"] == "rect")

    with open(txt_path, "w") as f:
        f.write("All Test Case Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset / model information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset family       : {cfg.dataset_family}\n")
        f.write(f"Dataset size         : {cfg.selected_dataset_size}\n")
        f.write(f"Output folder        : {cfg.outdir}\n")
        f.write(f"Dataset path         : {cfg.dataset_path}\n")
        f.write(f"Model path           : {cfg.model_path}\n")
        f.write("Branch input         : SDF only\n")
        f.write("Trunk input          : x, y, sdf(x,y)\n")
        f.write(f"Grid size            : {cfg.Nx} x {cfg.Ny}\n\n")

        f.write("Test set information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of test cases : {len(rows)}\n")
        f.write(f"Circle test cases    : {circle_count}\n")
        f.write(f"Rectangle test cases : {rect_count}\n")
        f.write(f"Full test loss       : {test_loss:.8e}\n\n")

        f.write("Mean error metrics over all test cases\n")
        f.write("-" * 70 + "\n")
        f.write(f"MSE mean             : {safe_mean(mse_values):.8e}\n")
        f.write(f"MAE mean             : {safe_mean(mae_values):.8e} K\n")
        f.write(f"MAPE mean            : {safe_mean(mape_values):.8e} %\n")
        f.write(f"RMSE mean            : {safe_mean(rmse_values):.8e} K\n")
        f.write(f"L2 error mean        : {safe_mean(l2_values):.8e}\n")
        f.write(f"Relative L2 mean     : {safe_mean(rel_l2_values):.8e}\n")
        f.write(f"Max abs error mean   : {safe_mean(max_abs_values):.8e} K\n")
        f.write(f"Max rel error mean   : {safe_mean(max_rel_values):.8e} %\n\n")

        f.write("Standard deviation over all test cases\n")
        f.write("-" * 70 + "\n")
        f.write(f"MSE std              : {safe_std(mse_values):.8e}\n")
        f.write(f"MAE std              : {safe_std(mae_values):.8e} K\n")
        f.write(f"MAPE std             : {safe_std(mape_values):.8e} %\n")
        f.write(f"RMSE std             : {safe_std(rmse_values):.8e} K\n")
        f.write(f"L2 error std         : {safe_std(l2_values):.8e}\n")
        f.write(f"Relative L2 std      : {safe_std(rel_l2_values):.8e}\n")
        f.write(f"Max abs error std    : {safe_std(max_abs_values):.8e} K\n")
        f.write(f"Max rel error std    : {safe_std(max_rel_values):.8e} %\n\n")

        f.write("Worst-case values over all test cases\n")
        f.write("-" * 70 + "\n")
        f.write(f"Worst MSE            : {safe_max(mse_values):.8e}\n")
        f.write(f"Worst MAE            : {safe_max(mae_values):.8e} K\n")
        f.write(f"Worst MAPE           : {safe_max(mape_values):.8e} %\n")
        f.write(f"Worst RMSE           : {safe_max(rmse_values):.8e} K\n")
        f.write(f"Worst L2 error       : {safe_max(l2_values):.8e}\n")
        f.write(f"Worst Relative L2    : {safe_max(rel_l2_values):.8e}\n")
        f.write(f"Worst max abs error  : {safe_max(max_abs_values):.8e} K\n")
        f.write(f"Worst max rel error  : {safe_max(max_rel_values):.8e} %\n\n")

        f.write("Runtime / memory\n")
        f.write("-" * 70 + "\n")
        f.write(f"Evaluation time [s]  : {elapsed_time_sec:.4f}\n")
        f.write(f"CPU memory [MB]      : {cpu_memory_mb:.4f}\n")
        f.write(f"GPU peak memory [MB] : {gpu_peak_memory_mb:.4f}\n\n")

        f.write("Saved detailed CSV\n")
        f.write("-" * 70 + "\n")
        f.write(csv_path + "\n")


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_all_test_cases(cfg):
    """
    Evaluate all unseen test cases.

    Nothing is printed to terminal.
    Results are saved in cfg.outdir.
    """

    os.makedirs(cfg.outdir, exist_ok=True)

    device = get_device()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    data = np.load(cfg.dataset_path, allow_pickle=True)

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

    coords = normalize_trunk_coordinates(
        coords,
        cfg.Lx,
        cfg.Ly
    )



    cfg.Nx = int(data["Nx"][0])
    cfg.Ny = int(data["Ny"][0])

    if "dataset_family" in data:
        cfg.dataset_family = str(data["dataset_family"][0])

    xgrid, ygrid = make_cartesian_grid_from_dataset(
        cfg.Lx,
        cfg.Ly,
        cfg.Nx,
        cfg.Ny
    )

    # ------------------------------------------------------------
    # Create test loader and split indices silently
    # ------------------------------------------------------------
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            _, _, test_loader, _, _, test_idx = make_loaders(
                sdf_all,
                mask_all,
                temp_all,
                cfg
            )

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------
    model = SimpleDeepONet(
        branch_in=sdf_all.shape[1],
        trunk_in=3,
        latent=cfg.latent_dim
    ).to(device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------
    # Full test-set masked MSE loss
    # ------------------------------------------------------------
    test_loss = 0.0

    with torch.no_grad():
        for sdf_b, mask_b, temp_b in test_loader:
            sdf_b = sdf_b.to(device)
            mask_b = mask_b.to(device)
            temp_b = temp_b.to(device)

            trunk_b = build_case_dependent_trunk_input(coords_t, sdf_b)
            pred = model(sdf_b, trunk_b)
            loss = masked_mse(pred, temp_b, mask_b)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    # ------------------------------------------------------------
    # Per-test-case metrics on actual FE mesh
    # ------------------------------------------------------------
    rows = []

    with torch.no_grad():
        for local_test_number, case_id in enumerate(test_idx):
            case_id = int(case_id)

            sdf_case = torch.tensor(
                sdf_all[case_id:case_id + 1],
                dtype=torch.float32,
                device=device
            )

            trunk_case = build_case_dependent_trunk_input(coords_t, sdf_case)

            pred_norm = model(sdf_case, trunk_case).cpu().numpy().reshape(
                cfg.Ny,
                cfg.Nx
            )

            pred_grid = pred_norm * (T_max - T_min) + T_min

            shape_code, p1, p2, cx, cy = meta_all[case_id]
            shape_name = "circle" if int(shape_code) == 0 else "rect"

            fe_data = np.load(str(fe_paths[case_id]))

            pts = fe_data["points"].astype(np.float64)
            true_fe = fe_data["temperature"].astype(np.float64)

            xv = pts[:, 0]
            yv = pts[:, 1]

            pred_fe = bilinear_interpolate_regular_grid(
                pred_grid,
                xgrid,
                ygrid,
                xv,
                yv
            )

            metrics = compute_metrics(true_fe, pred_fe)

            row = {
                "dataset_family": cfg.dataset_family,
                "dataset_size": cfg.selected_dataset_size,
                "test_case_number": int(local_test_number),
                "case_id": int(case_id),
                "shape": shape_name,
                "p1": float(p1),
                "p2": float(p2),
                "cx": float(cx),
                "cy": float(cy),
                **metrics,
            }

            rows.append(row)

    elapsed_time_sec = time.time() - start_time
    cpu_memory_mb = get_cpu_memory_mb()
    gpu_peak_memory_mb = get_gpu_peak_memory_mb()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    csv_path = os.path.join(cfg.outdir, "all_test_case_metrics.csv")
    txt_path = os.path.join(cfg.outdir, "all_test_case_metrics_summary.txt")

    save_metrics_csv(csv_path, rows)

    save_summary_txt(
        txt_path=txt_path,
        cfg=cfg,
        rows=rows,
        test_loss=test_loss,
        elapsed_time_sec=elapsed_time_sec,
        cpu_memory_mb=cpu_memory_mb,
        gpu_peak_memory_mb=gpu_peak_memory_mb,
        csv_path=csv_path,
    )

    save_rmse_mape_plot_and_csv(cfg, rows)
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_all_test_cases(Settings())
