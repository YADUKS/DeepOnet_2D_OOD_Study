"""
Minimal prediction script for one selected case.

This script:
1. Loads the saved dataset
2. Loads the saved best DeepONet model
3. Selects one case
4. Predicts temperature
5. Interpolates prediction onto the actual FE mesh
6. Plots:
   - Ground truth on FE mesh
   - Prediction on FE mesh
   - Signed error
   - Absolute error
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.config import Settings
from src.models.deeponet import SimpleDeepONet
from src.analysis.style import ERROR_CMAP, TEMPERATURE_CMAP
from src.fem.sampling import bilinear_interpolate_regular_grid


# ============================================================
# USER SETTINGS
# ============================================================

# Choose case selection mode:
# "test_index" -> select from unseen test set
# "case_id"    -> select original dataset case directly
SELECT_MODE = "test_index"

# Used only if SELECT_MODE = "test_index"
TEST_CASE_NUMBER = 0

# Used only if SELECT_MODE = "case_id"
ORIGINAL_CASE_ID = 100


# ============================================================
# SMALL HELPERS
# ============================================================

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    return device


def get_test_indices(ncases, split_seed):
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(ncases)

    n_train = int(0.8 * ncases)
    n_val = int(0.1 * ncases)

    test_idx = idx[n_train + n_val:]
    return test_idx


def select_case_id(ncases, split_seed):
    test_idx = get_test_indices(ncases, split_seed)

    if SELECT_MODE == "test_index":
        case_id = int(test_idx[TEST_CASE_NUMBER])

    elif SELECT_MODE == "case_id":
        case_id = int(ORIGINAL_CASE_ID)

    else:
        raise ValueError("SELECT_MODE must be 'test_index' or 'case_id'.")

    return case_id, test_idx


def add_colorbar(ax, im, label, fmt="{:.2f}", nticks=7):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)

    vmin, vmax = im.get_clim()
    ticks = np.linspace(vmin, vmax, nticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([fmt.format(t) for t in ticks])
    cbar.set_label(label)

    return cbar


def overlay_geometry(ax, shape_name, p1, p2, cx, cy, color="k", lw=1.8):
    if shape_name == "circle":
        r = p1 / 2.0
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        xb = cx + r * np.cos(theta)
        yb = cy + r * np.sin(theta)
        ax.plot(xb, yb, color=color, lw=lw)

    else:
        x1, y1 = cx - p1 / 2.0, cy - p2 / 2.0
        x2, y2 = cx + p1 / 2.0, cy - p2 / 2.0
        x3, y3 = cx + p1 / 2.0, cy + p2 / 2.0
        x4, y4 = cx - p1 / 2.0, cy + p2 / 2.0
        ax.plot(
            [x1, x2, x3, x4, x1],
            [y1, y2, y3, y4, y1],
            color=color,
            lw=lw
        )


def plot_fe_results(
    tri,
    true_fe,
    pred_fe,
    signed_fe,
    abs_fe,
    shape_name,
    p1,
    p2,
    cx,
    cy,
    Lx,
    Ly,
    case_id,
    save_path=None
):
    temp_min = np.nanmin(true_fe)
    temp_max = np.nanmax(true_fe)

    signed_min = np.nanmin(signed_fe)
    signed_max = np.nanmax(signed_fe)

    abs_min = np.nanmin(abs_fe)
    abs_max = np.nanmax(abs_fe)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    im0 = ax[0, 0].tripcolor(
        tri, true_fe, shading="flat", vmin=temp_min, vmax=temp_max,
        cmap=TEMPERATURE_CMAP,
    )
    overlay_geometry(ax[0, 0], shape_name, p1, p2, cx, cy, color="w")
    ax[0, 0].set_title(
        f"Ground truth on FE mesh\nMin={np.nanmin(true_fe):.2f} K, Max={np.nanmax(true_fe):.2f} K"
    )
    add_colorbar(ax[0, 0], im0, "Temperature [K]")

    im1 = ax[0, 1].tripcolor(
        tri, pred_fe, shading="flat", vmin=temp_min, vmax=temp_max,
        cmap=TEMPERATURE_CMAP,
    )
    overlay_geometry(ax[0, 1], shape_name, p1, p2, cx, cy, color="w")
    ax[0, 1].set_title(
        f"Predicted on FE mesh\nMin={np.nanmin(pred_fe):.2f} K, Max={np.nanmax(pred_fe):.2f} K"
    )
    add_colorbar(ax[0, 1], im1, "Temperature [K]")

    signed_limit = max(abs(float(signed_min)), abs(float(signed_max)), 1e-12)
    im2 = ax[1, 0].tripcolor(
        tri, signed_fe, shading="flat",
        vmin=-signed_limit, vmax=signed_limit, cmap=ERROR_CMAP,
    )
    overlay_geometry(ax[1, 0], shape_name, p1, p2, cx, cy, color="k")
    ax[1, 0].set_title(
        f"Signed error: Prediction - Truth\nMin={signed_min:.2f} K, Max={signed_max:.2f} K"
    )
    add_colorbar(ax[1, 0], im2, "Signed error [K]")

    im3 = ax[1, 1].tripcolor(
        tri, abs_fe, shading="flat", vmin=abs_min, vmax=abs_max,
        cmap=ERROR_CMAP,
    )
    overlay_geometry(ax[1, 1], shape_name, p1, p2, cx, cy, color="k")
    ax[1, 1].set_title(
        f"Absolute error\nMin={abs_min:.2f} K, Max={abs_max:.2f} K"
    )
    add_colorbar(ax[1, 1], im3, "Absolute error [K]")

    for a in ax.ravel():
        a.set_xlim(0.0, Lx)
        a.set_ylim(0.0, Ly)
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")

    fig.suptitle(f"DeepONet FE-mesh comparison | Case ID = {case_id}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Settings()
    device = get_device()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    data = np.load(cfg.dataset_path, allow_pickle=True)

    sdf_all = data["sdf"].astype(np.float32)
    temp_all = data["temp"].astype(np.float32)
    meta_all = data["meta"].astype(np.float32)
    coords = data["coords"].astype(np.float32)
    fe_paths = data["fe_paths"]

    T_min = float(data["T_min"][0])
    T_max = float(data["T_max"][0])

    Lx = float(data["Lx"][0])
    Ly = float(data["Ly"][0])
    Nx = int(data["Nx"][0])
    Ny = int(data["Ny"][0])

    ncases = sdf_all.shape[0]

    # ------------------------------------------------------------
    # Select case
    # ------------------------------------------------------------
    case_id, test_idx = select_case_id(ncases, cfg.split_seed)

    print("\nSelected case")
    print("-------------")
    print(f"Selection mode : {SELECT_MODE}")
    print(f"Selected case  : {case_id}")
    print(f"Test set size  : {len(test_idx)}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    model = SimpleDeepONet(
        branch_in=sdf_all.shape[1],
        latent=cfg.latent_dim
    ).to(device)

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

    print(f"\nLoaded model from: {cfg.model_path}")

    # ------------------------------------------------------------
    # Predict on Cartesian grid
    # ------------------------------------------------------------
    sdf_case = torch.tensor(
        sdf_all[case_id:case_id + 1],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        pred_norm = model(sdf_case, coords_t).cpu().numpy().reshape(Ny, Nx)

    pred_grid = pred_norm * (T_max - T_min) + T_min

    # ------------------------------------------------------------
    # Load actual FE mesh and FE ground truth
    # ------------------------------------------------------------
    fe_data = np.load(str(fe_paths[case_id]))

    pts = fe_data["points"].astype(np.float64)
    triangles = fe_data["triangles"].astype(np.int32)
    true_fe = fe_data["temperature"].astype(np.float64)

    # ------------------------------------------------------------
    # Interpolate predicted grid result onto FE mesh nodes
    # ------------------------------------------------------------
    xgrid = np.linspace(0.0, Lx, Nx)
    ygrid = np.linspace(0.0, Ly, Ny)

    xv = pts[:, 0]
    yv = pts[:, 1]

    pred_fe = bilinear_interpolate_regular_grid(
        pred_grid,
        xgrid,
        ygrid,
        xv,
        yv
    )

    signed_fe = pred_fe - true_fe
    abs_fe = np.abs(signed_fe)

    # ------------------------------------------------------------
    # FE triangulation
    # ------------------------------------------------------------
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)

    tri_mask = np.any(np.isnan(true_fe[triangles]), axis=1)
    tri.set_mask(tri_mask)

    # ------------------------------------------------------------
    # Geometry metadata
    # ------------------------------------------------------------
    shape_code, p1, p2, cx, cy = meta_all[case_id]
    shape_name = "circle" if int(shape_code) == 0 else "rect"

    # ------------------------------------------------------------
    # Error metrics on FE mesh
    # ------------------------------------------------------------
    valid = np.isfinite(true_fe) & np.isfinite(pred_fe)

    mae = np.mean(np.abs(signed_fe[valid]))
    rmse = np.sqrt(np.mean(signed_fe[valid] ** 2))
    rel_l2 = np.linalg.norm(signed_fe[valid]) / (np.linalg.norm(true_fe[valid]) + 1e-12)

    print("\nFE-mesh error summary")
    print("---------------------")
    print(f"Case ID : {case_id}")
    print(f"Shape   : {shape_name}")

    if shape_name == "circle":
        print(f"D       : {p1:.6f} m")
    else:
        print(f"W       : {p1:.6f} m")
        print(f"H       : {p2:.6f} m")

    print(f"cx      : {cx:.6f} m")
    print(f"cy      : {cy:.6f} m")
    print(f"MAE     : {mae:.6f} K")
    print(f"RMSE    : {rmse:.6f} K")
    print(f"Rel L2  : {rel_l2:.6e}")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    output_dir = "ngsolve_deeponet_dataset_withChoice"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"prediction_fe_case_{case_id:06d}.png"
    )

    plot_fe_results(
        tri=tri,
        true_fe=true_fe,
        pred_fe=pred_fe,
        signed_fe=signed_fe,
        abs_fe=abs_fe,
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        cx=cx,
        cy=cy,
        Lx=Lx,
        Ly=Ly,
        case_id=case_id,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
