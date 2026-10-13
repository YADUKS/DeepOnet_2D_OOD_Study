"""
Plotting helpers for preview and DeepONet testing.

This version provides:
    - preview plotting
    - default combined test-case plot
    - separate high-quality publishable FE figures
    - relative error figure
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.analysis.style import ERROR_CMAP, MASK_CMAP, SDF_CMAP, TEMPERATURE_CMAP
from src.fem.sampling import make_cartesian_grid


# ============================================================
# GENERIC HELPERS
# ============================================================

def add_full_height_colorbar(ax, mappable, label="", fmt="{:.2f}", vmin=None, vmax=None, nticks=7):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(mappable, cax=cax)

    if vmin is None or vmax is None:
        try:
            vmin_auto, vmax_auto = mappable.get_clim()
        except Exception:
            arr = np.asarray(mappable.get_array()).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                vmin_auto = np.min(arr)
                vmax_auto = np.max(arr)
            else:
                vmin_auto, vmax_auto = 0.0, 1.0

        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    ticks = np.linspace(vmin, vmax, nticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([fmt.format(t) for t in ticks])
    cbar.set_label(label)
    return cbar


def save_figure(fig, filename, dpi=300, save_pdf=False):
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {filename}")

    if save_pdf:
        pdf_name = os.path.splitext(filename)[0] + ".pdf"
        fig.savefig(pdf_name, bbox_inches="tight")
        print(f"Saved: {pdf_name}")

def get_test_case_output_dir(cfg, case_id):
    """
    Create a subfolder for all figures related to one selected test case.

    Folder format:
        cfg.outdir/test_case_number_XXXX_case_id_YYYYYY/

    where:
        XXXX   = cfg.test_case_number
        YYYYYY = actual dataset case ID
    """
    test_number = int(getattr(cfg, "test_case_number", 0))

    base_outdir = getattr(cfg, "active_test_outdir", cfg.outdir)

    case_dir = os.path.join(
        base_outdir,
        f"test_case_number_{test_number:04d}_case_id_{int(case_id):06d}"
    )

    os.makedirs(case_dir, exist_ok=True)
    return case_dir


def get_preview_output_dir(cfg):
    """
    Create a separate folder for preview figures.
    """
    preview_dir = os.path.join(cfg.outdir, "preview_figures")
    os.makedirs(preview_dir, exist_ok=True)
    return preview_dir

def style_axis(ax, cfg):
    ax.set_xlim(0.0, cfg.Lx)
    ax.set_ylim(0.0, cfg.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# ============================================================
# GEOMETRY DRAWING
# ============================================================

def rounded_rectangle_outline(cx, cy, W, H, R, n_corner_segments=48):
    radius = min(float(R), 0.5 * min(float(W), float(H)) - 1e-12)
    a = W / 2.0 - radius
    b = H / 2.0 - radius
    nseg = max(int(n_corner_segments), 4)

    corner_specs = [
        (cx + a, cy - b, -0.5 * np.pi, 0.0),
        (cx + a, cy + b, 0.0, 0.5 * np.pi),
        (cx - a, cy + b, 0.5 * np.pi, np.pi),
        (cx - a, cy - b, np.pi, 1.5 * np.pi),
    ]

    xvals = []
    yvals = []
    for x0, y0, theta0, theta1 in corner_specs:
        theta = np.linspace(theta0, theta1, nseg + 1)
        for t in theta:
            xvals.append(x0 + radius * np.cos(t))
            yvals.append(y0 + radius * np.sin(t))

    xvals.append(xvals[0])
    yvals.append(yvals[0])
    return np.asarray(xvals), np.asarray(yvals)


def plot_exact_geometry(ax, cfg, shape_name, p1, p2, cx, cy, p3=0.0):
    outer_x = [0.0, cfg.Lx, cfg.Lx, 0.0, 0.0]
    outer_y = [0.0, 0.0, cfg.Ly, cfg.Ly, 0.0]
    ax.plot(outer_x, outer_y, "k-", lw=2)

    if shape_name == "circle":
        r = p1 / 2.0
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        xb = cx + r * np.cos(theta)
        yb = cy + r * np.sin(theta)
        ax.plot(xb, yb, "r-", lw=2)
    elif shape_name == "rounded_rect":
        xb, yb = rounded_rectangle_outline(cx, cy, p1, p2, p3)
        ax.plot(xb, yb, "r-", lw=2)
    else:
        x1, y1 = cx - p1 / 2.0, cy - p2 / 2.0
        x2, y2 = cx + p1 / 2.0, cy - p2 / 2.0
        x3, y3 = cx + p1 / 2.0, cy + p2 / 2.0
        x4, y4 = cx - p1 / 2.0, cy + p2 / 2.0
        ax.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], "r-", lw=2)

    style_axis(ax, cfg)


def overlay_exact_geometry(ax, shape_name, p1, p2, cx, cy, color="w", lw=1.8, p3=0.0):
    if shape_name == "circle":
        r = p1 / 2.0
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        xb = cx + r * np.cos(theta)
        yb = cy + r * np.sin(theta)
        ax.plot(xb, yb, color=color, lw=lw)
    elif shape_name == "rounded_rect":
        xb, yb = rounded_rectangle_outline(cx, cy, p1, p2, p3)
        ax.plot(xb, yb, color=color, lw=lw)
    else:
        x1, y1 = cx - p1 / 2.0, cy - p2 / 2.0
        x2, y2 = cx + p1 / 2.0, cy - p2 / 2.0
        x3, y3 = cx + p1 / 2.0, cy + p2 / 2.0
        x4, y4 = cx - p1 / 2.0, cy + p2 / 2.0
        ax.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color=color, lw=lw)


# ============================================================
# PREVIEW
# ============================================================

def plot_preview(cfg, shape, p1, p2, cx, cy, sdf, mask, T, pts, triangles, T_fe):
    xgrid, ygrid, Xg, Yg, _ = make_cartesian_grid(cfg)

    print(f"Preview minimum temperature = {np.nanmin(T_fe):.4f} K")
    print(f"Preview maximum temperature = {np.nanmax(T_fe):.4f} K")

    fig, ax = plt.subplots(2, 3, figsize=(20, 11))

    plot_exact_geometry(ax[0, 0], cfg, shape, p1, p2, cx, cy)
    ax[0, 0].set_title("Actual geometry")

    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)
    ax[0, 1].triplot(tri, lw=0.5, color="tab:blue")
    overlay_exact_geometry(ax[0, 1], shape, p1, p2, cx, cy, color="r", lw=2)
    style_axis(ax[0, 1], cfg)
    ax[0, 1].set_title("Actual FE mesh")

    tri_mask = np.any(np.isnan(T_fe[triangles]), axis=1)
    tri.set_mask(tri_mask)
    im = ax[0, 2].tripcolor(
        tri, T_fe, shading="flat", cmap=TEMPERATURE_CMAP
    )
    overlay_exact_geometry(ax[0, 2], shape, p1, p2, cx, cy, color="w", lw=2)
    style_axis(ax[0, 2], cfg)
    ax[0, 2].set_title(f"Ground-truth FE solution\nMin={np.nanmin(T_fe):.2f} K, Max={np.nanmax(T_fe):.2f} K")
    add_full_height_colorbar(ax[0, 2], im, "Temperature [K]", vmin=np.nanmin(T_fe), vmax=np.nanmax(T_fe))

    im = ax[1, 0].contourf(Xg, Yg, sdf, 50, cmap=SDF_CMAP)
    ax[1, 0].contour(Xg, Yg, sdf, levels=[0.0], colors="k", linewidths=2)
    style_axis(ax[1, 0], cfg)
    ax[1, 0].set_title(f"SDF\nMin={np.nanmin(sdf):.5f}, Max={np.nanmax(sdf):.5f}")
    add_full_height_colorbar(ax[1, 0], im, "SDF", fmt="{:.5f}", vmin=np.nanmin(sdf), vmax=np.nanmax(sdf))

    im = ax[1, 1].imshow(
        mask,
        origin="lower",
        extent=[0, cfg.Lx, 0, cfg.Ly],
        interpolation="nearest",
        cmap=MASK_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    ax[1, 1].contour(Xg, Yg, sdf, levels=[0.0], colors="r", linewidths=2)
    style_axis(ax[1, 1], cfg)
    ax[1, 1].set_title("Binary mask + exact boundary")
    add_full_height_colorbar(ax[1, 1], im, "Mask", fmt="{:.0f}", vmin=0, vmax=1, nticks=2)

    im = ax[1, 2].contourf(
        Xg, Yg, T, 50, cmap=TEMPERATURE_CMAP
    )
    ax[1, 2].contour(Xg, Yg, sdf, levels=[0.0], colors="w", linewidths=1.5)
    style_axis(ax[1, 2], cfg)
    ax[1, 2].set_title(f"Masked sampled solution\nMin={np.nanmin(T):.2f} K, Max={np.nanmax(T):.2f} K")
    add_full_height_colorbar(ax[1, 2], im, "Temperature [K]", vmin=np.nanmin(T), vmax=np.nanmax(T))

    fig.suptitle(f"Preview case | Shape = {shape}", fontsize=15)
    plt.tight_layout()

    if cfg.save_figures:
        preview_dir = get_preview_output_dir(cfg)

        save_figure(
            fig,
            os.path.join(preview_dir, "preview_case_combined.png"),
            dpi=300,
            save_pdf=cfg.save_pdf_figures
        )

    if cfg.show_figures:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# COMBINED TEST CASE FIGURE
# ============================================================

def plot_test_case_combined(
    cfg,
    case_id,
    shape_name,
    p1,
    p2,
    cx,
    cy,
    tri,
    true_fe,
    pred_fe,
    true_masked,
    pred_masked,
    signed_fe,
    abs_fe,
    relative_fe,
    vmin,
    vmax,
    p3=0.0,
):
    _, _, Xg, Yg, _ = make_cartesian_grid(cfg)

    signed_lim = max(abs(np.nanmin(signed_fe)), abs(np.nanmax(signed_fe)))
    abs_min, abs_max = np.nanmin(abs_fe), np.nanmax(abs_fe)
    rel_min, rel_max = np.nanmin(relative_fe), np.nanmax(relative_fe)

    fig, ax = plt.subplots(3, 3, figsize=(18, 16))

    # 1
    plot_exact_geometry(ax[0, 0], cfg, shape_name, p1, p2, cx, cy, p3=p3)
    ax[0, 0].set_title(f"Exact geometry ({shape_name})")

    # 2
    ax[0, 1].triplot(tri, lw=0.5, color="tab:blue")
    overlay_exact_geometry(ax[0, 1], shape_name, p1, p2, cx, cy, color="r", lw=2, p3=p3)
    style_axis(ax[0, 1], cfg)
    ax[0, 1].set_title("Actual FE mesh")

    # 3 truth FE
    im = ax[0, 2].tripcolor(tri, true_fe, shading="flat", vmin=vmin, vmax=vmax, cmap=TEMPERATURE_CMAP)
    overlay_exact_geometry(ax[0, 2], shape_name, p1, p2, cx, cy, color="w", lw=2, p3=p3)
    style_axis(ax[0, 2], cfg)
    ax[0, 2].set_title(f"Ground truth on FE mesh\nMin={np.nanmin(true_fe):.2f} K, Max={np.nanmax(true_fe):.2f} K")
    add_full_height_colorbar(ax[0, 2], im, "Temperature [K]", vmin=vmin, vmax=vmax)

    # 4 pred FE
    im = ax[1, 0].tripcolor(tri, pred_fe, shading="flat", vmin=vmin, vmax=vmax, cmap=TEMPERATURE_CMAP)
    overlay_exact_geometry(ax[1, 0], shape_name, p1, p2, cx, cy, color="w", lw=2, p3=p3)
    style_axis(ax[1, 0], cfg)
    ax[1, 0].set_title(f"Predicted on FE mesh\nMin={np.nanmin(pred_fe):.2f} K, Max={np.nanmax(pred_fe):.2f} K")
    add_full_height_colorbar(ax[1, 0], im, "Temperature [K]", vmin=vmin, vmax=vmax)

    # 5 truth grid
    im = ax[1, 1].contourf(Xg, Yg, true_masked, 60, vmin=vmin, vmax=vmax, cmap=TEMPERATURE_CMAP)
    overlay_exact_geometry(ax[1, 1], shape_name, p1, p2, cx, cy, color="w", lw=2, p3=p3)
    style_axis(ax[1, 1], cfg)
    ax[1, 1].set_title(f"Ground truth on grid\nMin={np.nanmin(true_masked):.2f} K, Max={np.nanmax(true_masked):.2f} K")
    add_full_height_colorbar(ax[1, 1], im, "Temperature [K]", vmin=vmin, vmax=vmax)

    # 6 pred grid
    im = ax[1, 2].contourf(Xg, Yg, pred_masked, 60, vmin=vmin, vmax=vmax, cmap=TEMPERATURE_CMAP)
    overlay_exact_geometry(ax[1, 2], shape_name, p1, p2, cx, cy, color="w", lw=2, p3=p3)
    style_axis(ax[1, 2], cfg)
    ax[1, 2].set_title(f"Predicted on grid\nMin={np.nanmin(pred_masked):.2f} K, Max={np.nanmax(pred_masked):.2f} K")
    add_full_height_colorbar(ax[1, 2], im, "Temperature [K]", vmin=vmin, vmax=vmax)

    # 7 signed FE
    im = ax[2, 0].tripcolor(tri, signed_fe, shading="flat", vmin=-signed_lim, vmax=signed_lim, cmap=ERROR_CMAP)
    overlay_exact_geometry(ax[2, 0], shape_name, p1, p2, cx, cy, color="k", lw=2, p3=p3)
    style_axis(ax[2, 0], cfg)
    ax[2, 0].set_title(f"Signed error on FE mesh\nMin={np.nanmin(signed_fe):.2f} K, Max={np.nanmax(signed_fe):.2f} K")
    add_full_height_colorbar(ax[2, 0], im, "Signed error [K]", vmin=-signed_lim, vmax=signed_lim)

    # 8 absolute FE
    im = ax[2, 1].tripcolor(tri, abs_fe, shading="flat", vmin=abs_min, vmax=abs_max, cmap=ERROR_CMAP)
    overlay_exact_geometry(ax[2, 1], shape_name, p1, p2, cx, cy, color="k", lw=2, p3=p3)
    style_axis(ax[2, 1], cfg)
    ax[2, 1].set_title(f"Absolute error on FE mesh\nMin={abs_min:.2f} K, Max={abs_max:.2f} K")
    add_full_height_colorbar(ax[2, 1], im, "Absolute error [K]", vmin=abs_min, vmax=abs_max)

    # 9 relative FE
    im = ax[2, 2].tripcolor(tri, relative_fe, shading="flat", vmin=rel_min, vmax=rel_max, cmap=ERROR_CMAP)
    overlay_exact_geometry(ax[2, 2], shape_name, p1, p2, cx, cy, color="k", lw=2, p3=p3)
    style_axis(ax[2, 2], cfg)
    ax[2, 2].set_title(f"Relative error on FE mesh\nMin={rel_min:.2f} %, Max={rel_max:.2f} %")
    add_full_height_colorbar(ax[2, 2], im, "Relative error [%]", vmin=rel_min, vmax=rel_max)

    fig.suptitle(f"Test case comparison | Case ID = {case_id} | Shape = {shape_name}", fontsize=16)
    plt.tight_layout()

    if cfg.save_figures:
        case_dir = get_test_case_output_dir(cfg, case_id)

        combined_path = os.path.join(
            case_dir,
            f"test_case_number_{int(cfg.test_case_number):04d}_case_id_{int(case_id):06d}_combined.png"
        )

        save_figure(
            fig,
            combined_path,
            dpi=300,
            save_pdf=cfg.save_pdf_figures
        )

    if cfg.show_figures:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# INDIVIDUAL PUBLISHABLE FIGURES
# ============================================================

def save_publishable_single_fe_plot(
    cfg,
    output_dir,
    filename,
    tri,
    values,
    title="",
    colorbar_label="",
    shape_name="circle",
    p1=0.0,
    p2=0.0,
    p3=0.0,
    cx=0.0,
    cy=0.0,
    vmin=None,
    vmax=None,
    cmap=TEMPERATURE_CMAP,
    overlay_color="w",
    fmt="{:.2f}",
):
    """
    Save one high-quality FE-mesh figure inside the selected test-case folder.

    Saves:
        PNG at cfg.publish_dpi
        PDF if cfg.save_pdf_figures = True
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.tripcolor(
        tri,
        values,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    overlay_exact_geometry(
        ax,
        shape_name,
        p1,
        p2,
        cx,
        cy,
        color=overlay_color,
        lw=2,
        p3=p3
    )

    if title:
        ax.set_title(title, fontsize=13)

    style_axis(ax, cfg)

    add_full_height_colorbar(
        ax,
        im,
        label=colorbar_label,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax
    )

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(output_dir, filename),
        dpi=cfg.publish_dpi,
        save_pdf=cfg.save_pdf_figures
    )

    plt.close(fig)


def save_test_case_publishable_figures(
    cfg,
    case_id,
    shape_name,
    p1,
    p2,
    cx,
    cy,
    tri,
    true_fe,
    pred_fe,
    signed_fe,
    abs_fe,
    relative_fe,
    vmin,
    vmax,
    p3=0.0,
):
    signed_lim = max(abs(np.nanmin(signed_fe)), abs(np.nanmax(signed_fe)))
    abs_min, abs_max = np.nanmin(abs_fe), np.nanmax(abs_fe)
    rel_min, rel_max = np.nanmin(relative_fe), np.nanmax(relative_fe)

    prefix = f"test_case_number_{int(cfg.test_case_number):04d}_case_id_{int(case_id):06d}"

    case_dir = get_test_case_output_dir(cfg, case_id)

    save_publishable_single_fe_plot(
        cfg=cfg,
        output_dir=case_dir,
        filename=f"{prefix}_ground_truth_fe.png",
        tri=tri,
        values=true_fe,
        title="",
        colorbar_label="Temperature [K]",
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        p3=p3,
        cx=cx,
        cy=cy,
        vmin=vmin,
        vmax=vmax,
        cmap=TEMPERATURE_CMAP,
        overlay_color="w",
    )

    save_publishable_single_fe_plot(
        cfg=cfg,
        output_dir=case_dir,
        filename=f"{prefix}_predicted_fe.png",
        tri=tri,
        values=pred_fe,
        title="",
        colorbar_label="Temperature [K]",
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        p3=p3,
        cx=cx,
        cy=cy,
        vmin=vmin,
        vmax=vmax,
        cmap=TEMPERATURE_CMAP,
        overlay_color="w",
    )

    save_publishable_single_fe_plot(
        cfg=cfg,
        output_dir=case_dir,
        filename=f"{prefix}_signed_error_fe.png",
        tri=tri,
        values=signed_fe,
        title="",
        colorbar_label="Signed error [K]",
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        p3=p3,
        cx=cx,
        cy=cy,
        vmin=-signed_lim,
        vmax=signed_lim,
        cmap=ERROR_CMAP,
        overlay_color="k",
    )

    save_publishable_single_fe_plot(
        cfg=cfg,
        output_dir=case_dir,
        filename=f"{prefix}_absolute_error_fe.png",
        tri=tri,
        values=abs_fe,
        title="",
        colorbar_label="Absolute error [K]",
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        p3=p3,
        cx=cx,
        cy=cy,
        vmin=abs_min,
        vmax=abs_max,
        cmap=ERROR_CMAP,
        overlay_color="k",
    )

    save_publishable_single_fe_plot(
        cfg=cfg,
        output_dir=case_dir,
        filename=f"{prefix}_relative_error_fe.png",
        tri=tri,
        values=relative_fe,
        title="",
        colorbar_label="Relative error [%]",
        shape_name=shape_name,
        p1=p1,
        p2=p2,
        p3=p3,
        cx=cx,
        cy=cy,
        vmin=rel_min,
        vmax=rel_max,
        cmap=ERROR_CMAP,
        overlay_color="k",
    )
