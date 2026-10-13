"""
Plot the sampling space used in the generated dataset.

This script reads dataset.npz and plots the actual sampled geometry
parameters stored in the metadata.

Dataset metadata format:
    meta = [shape_code, p1, p2, cx, cy]

For circle:
    shape_code = 0
    p1 = D
    p2 = 0
    cx = circle centre x
    cy = circle centre y

For rectangle:
    shape_code = 1
    p1 = W
    p2 = H
    cx = rectangle centre x
    cy = rectangle centre y
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Settings


# ============================================================
# USER OPTION
# ============================================================

SAVE_FIGURE = True
SHOW_FIGURE = True
DPI = 600


# ============================================================
# HELPER
# ============================================================

def save_and_show(fig, save_path):
    if SAVE_FIGURE:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def plot_plate_with_samples(cfg, cx, cy, values, title, colorbar_label, save_path):
    """
    Plot sampled centre locations inside the plate.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(cx, cy, c=values, s=18, edgecolors="none")

    ax.plot(
        [0.0, cfg.Lx, cfg.Lx, 0.0, 0.0],
        [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0],
        "k-",
        linewidth=2,
    )

    ax.set_xlim(0.0, cfg.Lx)
    ax.set_ylim(0.0, cfg.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("Centre x-coordinate, cx [m]")
    ax.set_ylabel("Centre y-coordinate, cy [m]")
    ax.set_title(title)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    save_and_show(fig, save_path)


# ============================================================
# CIRCLE PLOTS
# ============================================================

def plot_circle_sampling(cfg, meta, output_dir):
    """
    Plot sampling space for circle-only or circle subset of combined dataset.
    """
    D = meta[:, 1]
    cx = meta[:, 3]
    cy = meta[:, 4]

    print("\nCircle sampling summary")
    print("-----------------------")
    print(f"Number of circle samples : {len(D)}")
    print(f"D  : min = {D.min():.6f}, max = {D.max():.6f}")
    print(f"cx : min = {cx.min():.6f}, max = {cx.max():.6f}")
    print(f"cy : min = {cy.min():.6f}, max = {cy.max():.6f}")

    # ------------------------------------------------------------
    # Combined 2 x 2 sampling plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    sc1 = ax[0, 0].scatter(D, cx, c=cy, s=18)
    ax[0, 0].set_xlabel("Diameter D [m]")
    ax[0, 0].set_ylabel("Centre x-coordinate cx [m]")
    ax[0, 0].set_title("Circle sampling: D vs cx")
    plt.colorbar(sc1, ax=ax[0, 0], label="cy [m]")

    sc2 = ax[0, 1].scatter(D, cy, c=cx, s=18)
    ax[0, 1].set_xlabel("Diameter D [m]")
    ax[0, 1].set_ylabel("Centre y-coordinate cy [m]")
    ax[0, 1].set_title("Circle sampling: D vs cy")
    plt.colorbar(sc2, ax=ax[0, 1], label="cx [m]")

    sc3 = ax[1, 0].scatter(cx, cy, c=D, s=18)
    ax[1, 0].plot(
        [0.0, cfg.Lx, cfg.Lx, 0.0, 0.0],
        [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0],
        "k-",
        linewidth=1.5,
    )
    ax[1, 0].set_xlim(0.0, cfg.Lx)
    ax[1, 0].set_ylim(0.0, cfg.Ly)
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlabel("Centre x-coordinate cx [m]")
    ax[1, 0].set_ylabel("Centre y-coordinate cy [m]")
    ax[1, 0].set_title("Circle centre locations")
    plt.colorbar(sc3, ax=ax[1, 0], label="D [m]")

    ax[1, 1].hist(D, bins=30)
    ax[1, 1].set_xlabel("Diameter D [m]")
    ax[1, 1].set_ylabel("Count")
    ax[1, 1].set_title("Distribution of D")

    fig.suptitle("Circle sampling space", fontsize=15)
    plt.tight_layout()

    save_and_show(
        fig,
        os.path.join(output_dir, "sampling_space_circle_combined.png")
    )

    # ------------------------------------------------------------
    # Separate publishable centre-location plot
    # ------------------------------------------------------------
    plot_plate_with_samples(
        cfg=cfg,
        cx=cx,
        cy=cy,
        values=D,
        title="Circle centre sampling space",
        colorbar_label="Diameter D [m]",
        save_path=os.path.join(output_dir, "sampling_space_circle_centres.png"),
    )


# ============================================================
# RECTANGLE PLOTS
# ============================================================

def plot_rect_sampling(cfg, meta, output_dir):
    """
    Plot sampling space for rectangle-only or rectangle subset of combined dataset.
    """
    W = meta[:, 1]
    H = meta[:, 2]
    cx = meta[:, 3]
    cy = meta[:, 4]

    print("\nRectangle sampling summary")
    print("--------------------------")
    print(f"Number of rectangle samples : {len(W)}")
    print(f"W  : min = {W.min():.6f}, max = {W.max():.6f}")
    print(f"H  : min = {H.min():.6f}, max = {H.max():.6f}")
    print(f"cx : min = {cx.min():.6f}, max = {cx.max():.6f}")
    print(f"cy : min = {cy.min():.6f}, max = {cy.max():.6f}")

    # ------------------------------------------------------------
    # Combined 2 x 2 sampling plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    sc1 = ax[0, 0].scatter(W, H, c=cx, s=18)
    ax[0, 0].set_xlabel("Width W [m]")
    ax[0, 0].set_ylabel("Height H [m]")
    ax[0, 0].set_title("Rectangle sampling: W vs H")
    plt.colorbar(sc1, ax=ax[0, 0], label="cx [m]")

    sc2 = ax[0, 1].scatter(W, cx, c=cy, s=18)
    ax[0, 1].set_xlabel("Width W [m]")
    ax[0, 1].set_ylabel("Centre x-coordinate cx [m]")
    ax[0, 1].set_title("Rectangle sampling: W vs cx")
    plt.colorbar(sc2, ax=ax[0, 1], label="cy [m]")

    sc3 = ax[1, 0].scatter(cx, cy, c=W, s=18)
    ax[1, 0].plot(
        [0.0, cfg.Lx, cfg.Lx, 0.0, 0.0],
        [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0],
        "k-",
        linewidth=1.5,
    )
    ax[1, 0].set_xlim(0.0, cfg.Lx)
    ax[1, 0].set_ylim(0.0, cfg.Ly)
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlabel("Centre x-coordinate cx [m]")
    ax[1, 0].set_ylabel("Centre y-coordinate cy [m]")
    ax[1, 0].set_title("Rectangle centre locations")
    plt.colorbar(sc3, ax=ax[1, 0], label="W [m]")

    ax[1, 1].hist(W, bins=30, alpha=0.7, label="W")
    ax[1, 1].hist(H, bins=30, alpha=0.7, label="H")
    ax[1, 1].set_xlabel("Dimension [m]")
    ax[1, 1].set_ylabel("Count")
    ax[1, 1].set_title("Distribution of W and H")
    ax[1, 1].legend()

    fig.suptitle("Rectangle sampling space", fontsize=15)
    plt.tight_layout()

    save_and_show(
        fig,
        os.path.join(output_dir, "sampling_space_rect_combined.png")
    )

    # ------------------------------------------------------------
    # Separate publishable centre-location plot
    # ------------------------------------------------------------
    plot_plate_with_samples(
        cfg=cfg,
        cx=cx,
        cy=cy,
        values=W,
        title="Rectangle centre sampling space",
        colorbar_label="Width W [m]",
        save_path=os.path.join(output_dir, "sampling_space_rect_centres.png"),
    )


# ============================================================
# COMBINED FAMILY PLOT
# ============================================================

def plot_combined_shape_distribution(cfg, meta, output_dir):
    """
    Plot circle and rectangle centre locations together.
    """
    shape_code = meta[:, 0]
    circle_meta = meta[shape_code == 0]
    rect_meta = meta[shape_code == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(circle_meta) > 0:
        ax.scatter(
            circle_meta[:, 3],
            circle_meta[:, 4],
            s=18,
            label="Circle",
            alpha=0.8,
        )

    if len(rect_meta) > 0:
        ax.scatter(
            rect_meta[:, 3],
            rect_meta[:, 4],
            s=18,
            marker="s",
            label="Rectangle",
            alpha=0.8,
        )

    ax.plot(
        [0.0, cfg.Lx, cfg.Lx, 0.0, 0.0],
        [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0],
        "k-",
        linewidth=2,
    )

    ax.set_xlim(0.0, cfg.Lx)
    ax.set_ylim(0.0, cfg.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("Centre x-coordinate cx [m]")
    ax.set_ylabel("Centre y-coordinate cy [m]")
    ax.set_title("Combined dataset centre locations")
    ax.legend()

    plt.tight_layout()

    save_and_show(
        fig,
        os.path.join(output_dir, "sampling_space_combined_centres.png")
    )


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Settings()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    data = np.load(cfg.dataset_path, allow_pickle=True)

    meta = data["meta"].astype(np.float32)

    if "dataset_family" in data:
        dataset_family = str(data["dataset_family"][0])
    elif "shape_mode" in data:
        dataset_family = str(data["shape_mode"][0])
    else:
        dataset_family = cfg.dataset_family

    # Keep cfg geometry consistent with saved dataset
    cfg.Lx = float(data["Lx"][0])
    cfg.Ly = float(data["Ly"][0])
    cfg.Nx = int(data["Nx"][0])
    cfg.Ny = int(data["Ny"][0])
    cfg.dataset_family = dataset_family

    output_dir = cfg.outdir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded dataset from : {cfg.dataset_path}")
    print(f"Dataset family     : {dataset_family}")
    print(f"Number of samples  : {meta.shape[0]}")
    print(f"Output directory   : {output_dir}")

    # ------------------------------------------------------------
    # Separate circle and rectangle data
    # ------------------------------------------------------------
    shape_code = meta[:, 0]
    circle_meta = meta[shape_code == 0]
    rect_meta = meta[shape_code == 1]

    print("\nShape counts")
    print("------------")
    print(f"Circle samples    : {len(circle_meta)}")
    print(f"Rectangle samples : {len(rect_meta)}")

    # ------------------------------------------------------------
    # Plot according to available shape types
    # ------------------------------------------------------------
    if len(circle_meta) > 0:
        plot_circle_sampling(cfg, circle_meta, output_dir)

    if len(rect_meta) > 0:
        plot_rect_sampling(cfg, rect_meta, output_dir)

    if len(circle_meta) > 0 and len(rect_meta) > 0:
        plot_combined_shape_distribution(cfg, meta, output_dir)


if __name__ == "__main__":
    main()