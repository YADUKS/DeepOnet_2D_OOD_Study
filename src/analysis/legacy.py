"""
Evaluate a trained DeepONet model on one selected test case.

Outputs:
    - default combined test-case figure
    - separate high-quality publishable FE figures
    - metrics CSV
"""

import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pyexpat import model
import numpy as np
import torch
import matplotlib.tri as mtri

try:
    import psutil
except ImportError:
    psutil = None

from src.config import Settings
from src.data.dataset import make_loaders
from src.models.deeponet import SimpleDeepONet, masked_mse
from src.fem.sampling import make_cartesian_grid, bilinear_interpolate_regular_grid
from src.analysis.plots import (
    plot_test_case_combined,
    save_test_case_publishable_figures,
    get_test_case_output_dir,
)


# ============================================================
# SYSTEM HELPERS
# ============================================================

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
    trunk_b = torch.cat([coords_batch, sdf_feature], dim=-1)
    return trunk_b


def write_test_case_summary_txt(
    cfg,
    case_id,
    shape_name,
    p1,
    p2,
    cx,
    cy,
    test_loss,
    mse,
    mae,
    mape,
    rmse,
    l2_error,
    relative_l2,
    memory_used_mb,
    cpu_memory_mb,
    gpu_peak_memory_mb,
):
    """
    Save the selected test-case summary into the same subfolder
    where the test-case figures are saved.
    """

    case_dir = get_test_case_output_dir(cfg, case_id)

    summary_path = os.path.join(
        case_dir,
        f"test_case_number_{int(cfg.test_case_number):04d}_case_id_{int(case_id):06d}_summary.txt"
    )

    with open(summary_path, "w") as f:
        f.write("DeepONet Test Case Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset family       : {cfg.dataset_family}\n")
        f.write(f"Dataset size         : {cfg.selected_dataset_size}\n")
        f.write(f"Dataset folder       : {cfg.outdir}\n")
        f.write(f"Dataset path         : {cfg.dataset_path}\n")
        f.write(f"Model path           : {cfg.model_path}\n\n")

        f.write("Selected test case\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test case number     : {int(cfg.test_case_number)}\n")
        f.write(f"Actual case ID       : {int(case_id)}\n")
        f.write(f"Shape                : {shape_name}\n")

        if shape_name == "circle":
            f.write(f"Diameter D           : {p1:.8f} m\n")
        else:
            f.write(f"Width W              : {p1:.8f} m\n")
            f.write(f"Height H             : {p2:.8f} m\n")

        f.write(f"Centre cx            : {cx:.8f} m\n")
        f.write(f"Centre cy            : {cy:.8f} m\n\n")

        f.write("Model input/output description\n")
        f.write("-" * 70 + "\n")
        f.write("Branch input         : SDF only\n")
        f.write("Trunk input          : x, y\n")
        f.write("Output               : Normalized temperature field\n")
        f.write("Evaluation field     : Prediction interpolated onto actual FE mesh\n\n")

        f.write("Error metrics on FE mesh\n")
        f.write("-" * 70 + "\n")
        f.write(f"Full test loss       : {test_loss:.8e}\n")
        f.write(f"MSE                  : {mse:.8e}\n")
        f.write(f"MAE                  : {mae:.8e} K\n")
        f.write(f"MAPE                 : {mape:.8e} %\n")
        f.write(f"RMSE                 : {rmse:.8e} K\n")
        f.write(f"L2 error             : {l2_error:.8e}\n")
        f.write(f"Relative L2 error    : {relative_l2:.8e}\n\n")

        f.write("Memory information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Memory used          : {memory_used_mb:.4f} MB\n")
        f.write(f"CPU memory used      : {cpu_memory_mb:.4f} MB\n")
        f.write(f"GPU peak memory used : {gpu_peak_memory_mb:.4f} MB\n\n")

        f.write("Saved figure folder\n")
        f.write("-" * 70 + "\n")
        f.write(f"{case_dir}\n")

    print(f"Saved test-case summary text file to: {summary_path}")

    return summary_path

def get_cpu_memory_mb():
    if psutil is None:
        return np.nan
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return np.nan


def save_metrics_csv(cfg, metrics_dict):
    with open(cfg.metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics_dict.keys()))
        writer.writerow(list(metrics_dict.values()))
    print(f"Saved: {cfg.metrics_csv_path}")


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_model(cfg):
    device = get_device()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

    xgrid, ygrid, _, _, _ = make_cartesian_grid(cfg)

    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = make_loaders(
        sdf_all, mask_all, temp_all, cfg
    )

    if cfg.test_case_number >= len(test_idx):
        raise ValueError(
            f"cfg.test_case_number = {cfg.test_case_number} is too large. "
            f"Available test cases = {len(test_idx)}"
        )

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    model = SimpleDeepONet(
        branch_in=sdf_all.shape[1],
        trunk_in=3,
        latent=cfg.latent_dim
    ).to(device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------
    # FULL TEST LOSS
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
    print(f"Final Test Loss = {test_loss:.6e}")

    # ------------------------------------------------------------
    # SELECTED TEST CASE
    # ------------------------------------------------------------
    case_id = int(test_idx[cfg.test_case_number])

    sdf_case = torch.tensor(
        sdf_all[case_id:case_id + 1],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        trunk_case = build_case_dependent_trunk_input(coords_t, sdf_case)
        pred_norm = model(sdf_case, trunk_case).cpu().numpy().reshape(cfg.Ny, cfg.Nx)
        

    true_norm = temp_all[case_id].reshape(cfg.Ny, cfg.Nx)
    mask = mask_all[case_id].reshape(cfg.Ny, cfg.Nx)

    pred_grid = pred_norm * (T_max - T_min) + T_min
    true_grid = true_norm * (T_max - T_min) + T_min

    shape_code, p1, p2, cx, cy = meta_all[case_id]
    shape_name = "circle" if int(shape_code) == 0 else "rect"

    # ------------------------------------------------------------
    # LOAD FE GROUND TRUTH
    # ------------------------------------------------------------
    fe_data = np.load(str(fe_paths[case_id]))
    pts = fe_data["points"].astype(np.float64)
    triangles = fe_data["triangles"].astype(np.int32)
    true_fe = fe_data["temperature"].astype(np.float64)

    xv = pts[:, 0]
    yv = pts[:, 1]

    pred_fe = bilinear_interpolate_regular_grid(pred_grid, xgrid, ygrid, xv, yv)

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

    vmin = np.nanmin(true_fe)
    vmax = np.nanmax(true_fe)

    # ------------------------------------------------------------
    # SAVE COMBINED + PUBLISHABLE FIGURES
    # ------------------------------------------------------------
    plot_test_case_combined(
        cfg=cfg,
        case_id=case_id,
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        cx=cx,
        cy=cy,
        tri=tri,
        true_fe=true_fe,
        pred_fe=pred_fe,
        true_masked=true_masked,
        pred_masked=pred_masked,
        signed_fe=signed_fe,
        abs_fe=abs_fe,
        relative_fe=relative_fe,
        vmin=vmin,
        vmax=vmax,
    )

    save_test_case_publishable_figures(
        cfg=cfg,
        case_id=case_id,
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        cx=cx,
        cy=cy,
        tri=tri,
        true_fe=true_fe,
        pred_fe=pred_fe,
        signed_fe=signed_fe,
        abs_fe=abs_fe,
        relative_fe=relative_fe,
        vmin=vmin,
        vmax=vmax,
    )

    # ------------------------------------------------------------
    # METRICS ON FE MESH
    # ------------------------------------------------------------
    valid = np.isfinite(true_fe) & np.isfinite(pred_fe)

    diff = pred_fe[valid] - true_fe[valid]
    true_valid = true_fe[valid]

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(100.0 * np.abs(diff) / (np.abs(true_valid) + 1e-12)))
    rmse = float(np.sqrt(mse))
    l2_error = float(np.linalg.norm(diff))
    relative_l2 = float(l2_error / (np.linalg.norm(true_valid) + 1e-12))

    cpu_memory_mb = get_cpu_memory_mb()
    gpu_peak_memory_mb = get_gpu_peak_memory_mb()

    memory_used_mb = gpu_peak_memory_mb if torch.cuda.is_available() else cpu_memory_mb

    

    metrics = {
        "dataset_family": cfg.dataset_family,
        "dataset_size": cfg.selected_dataset_size,
        "case_id": case_id,
        "shape_name": shape_name,
        "mse": mse,
        "mae": mae,
        "mape_percent": mape,
        "rmse": rmse,
        "l2_error": l2_error,
        "relative_l2": relative_l2,
        "memory_used_mb": memory_used_mb,
        "cpu_memory_mb": cpu_memory_mb,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "test_loss": test_loss,
    }

    save_metrics_csv(cfg, metrics)

    write_test_case_summary_txt(
        cfg=cfg,
        case_id=case_id,
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        cx=cx,
        cy=cy,
        test_loss=test_loss,
        mse=mse,
        mae=mae,
        mape=mape,
        rmse=rmse,
        l2_error=l2_error,
        relative_l2=relative_l2,
        memory_used_mb=memory_used_mb,
        cpu_memory_mb=cpu_memory_mb,
        gpu_peak_memory_mb=gpu_peak_memory_mb,
    )

    print("\nTest case summary")
    print("-----------------")
    print(f"Dataset family : {cfg.dataset_family}")
    print(f"Dataset size   : {cfg.selected_dataset_size}")
    print(f"Case ID        : {case_id}")
    print(f"Shape          : {shape_name}")
    if shape_name == "circle":
        print(f"D              : {p1:.6f}")
    else:
        print(f"W              : {p1:.6f}")
        print(f"H              : {p2:.6f}")
    print(f"cx             : {cx:.6f}")
    print(f"cy             : {cy:.6f}")
    print(f"MSE            : {mse:.6e}")
    print(f"MAE            : {mae:.6e} K")
    print(f"MAPE           : {mape:.6e} %")
    print(f"RMSE           : {rmse:.6e} K")
    print(f"L2 error       : {l2_error:.6e}")
    print(f"Relative L2    : {relative_l2:.6e}")
    print(f"Memory used    : {memory_used_mb:.4f} MB")

    return metrics


if __name__ == "__main__":
    evaluate_model(Settings())
