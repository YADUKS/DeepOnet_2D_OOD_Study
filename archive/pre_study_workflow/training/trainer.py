"""
Train DeepONet using the generated dataset.

Model:
    - branch input = full SDF field
    - trunk input  = normalized x, normalized y, and SDF(x,y)

Outputs:
    - best model
    - train_loss_history.csv
    - val_loss_history.csv
    - train_val_loss.png
    - training_summary.txt
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import json
import platform
import sys
import time
import random
from dataclasses import asdict
from importlib import metadata
import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import psutil
except ImportError:
    psutil = None

from src.config import Settings
from src.data.design import relative_path, file_digest
from src.data.dataset import make_loaders
from src.models.deeponet import SimpleDeepONet, masked_mse
from src.data.protocol import (
    load_and_prepare_training_dataset,
    resolve_training_dataset_path,
    validate_model_input_contract,
)


# ============================================================
# SYSTEM HELPERS
# ============================================================

from src.training.runtime import get_device


def set_deterministic_training(seed, deterministic=True):
    """
    Seed Python, NumPy, and PyTorch so repeated training runs use the same
    initialization, batch order, and CUDA deterministic modes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def get_package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_environment_info():
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "none",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "ngsolve": get_package_version("ngsolve"),
        "netgen": get_package_version("netgen"),
    }


def save_config_snapshot(cfg):
    training_dataset_path = resolve_training_dataset_path(cfg)
    payload = {
        "settings": asdict(cfg),
        "resolved_paths": {
            "outdir": cfg.outdir,
            "dataset_path": training_dataset_path,
            "model_path": cfg.model_path,
            "split_indices_path": cfg.split_indices_path,
            "temperature_normalization_path": cfg.temperature_normalization_path,
        },
        "environment": collect_environment_info(),
    }

    with open(cfg.config_used_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {cfg.config_used_path}")


def save_split_indices(
    cfg,
    train_idx,
    val_idx,
    test_idx,
    split_method,
    family_split_counts,
):
    np.savez(
        cfg.split_indices_path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        split_seed=np.array([cfg.split_seed], dtype=np.int64),
        split_method=np.array([split_method], dtype=object),
        family_split_counts_json=np.array(
            [json.dumps(family_split_counts, sort_keys=True)],
            dtype=object,
        ),
    )
    print(f"Saved: {cfg.split_indices_path}")


def save_temperature_normalization(cfg, prepared_data):
    """Save the exact scaler fitted for this training run."""
    fitted_on = (
        "training_split_solid_points"
        if prepared_data["mode"] == "composed"
        else "legacy_dataset_precomputed"
    )
    np.savez(
        cfg.temperature_normalization_path,
        method=np.array([prepared_data["normalization_method"]], dtype=object),
        fitted_on=np.array([fitted_on], dtype=object),
        temperature_min=np.array(
            [prepared_data["temperature_min"]],
            dtype=object,
        ),
        temperature_max=np.array(
            [prepared_data["temperature_max"]],
            dtype=object,
        ),
        split_seed=np.array([cfg.split_seed], dtype=np.int64),
        training_dataset_path=np.array(
            [relative_path(prepared_data["dataset_path"], cfg.temperature_normalization_path)],
            dtype=object,
        ),
        training_dataset_sha256=np.array([file_digest(prepared_data["dataset_path"])]),
    )
    print(f"Saved: {cfg.temperature_normalization_path}")


def normalize_trunk_coordinates(coords, Lx, Ly):
    """
    Normalize trunk coordinates.

    Original:
        x in [0, Lx]
        y in [0, Ly]

    Normalized:
        x_norm = x / Lx
        y_norm = y / Ly

    Parameters
    ----------
    coords : np.ndarray
        Shape: [n_points, 2]

    Lx, Ly : float
        Domain length and height.

    Returns
    -------
    coords_norm : np.ndarray
        Shape: [n_points, 2]
    """

    coords_norm = coords.copy()

    coords_norm[:, 0] = coords_norm[:, 0] / Lx
    coords_norm[:, 1] = coords_norm[:, 1] / Ly

    return coords_norm.astype(np.float32)



def build_case_dependent_trunk_input(coords_t, sdf_b):
    """
    Build trunk input for each case in the batch.

    Current trunk input:
        [x, y, sdf(x,y)]

    Parameters
    ----------
    coords_t : tensor
        Shape: [n_points, 2]

    sdf_b : tensor
        Shape: [batch_size, n_points]

    Returns
    -------
    trunk_b : tensor
        Shape: [batch_size, n_points, 3]
    """

    if coords_t.dim() != 2 or coords_t.shape[1] != 2:
        raise ValueError("coords_t must contain normalized [x, y] pairs.")
    if sdf_b.dim() != 2:
        raise ValueError("sdf_b must contain one complete flattened SDF per case.")
    if sdf_b.shape[1] != coords_t.shape[0]:
        raise ValueError(
            "SDF point count does not match the normalized coordinate grid."
        )

    batch_size = sdf_b.shape[0]

    coords_batch = coords_t.unsqueeze(0).expand(batch_size, -1, -1)
    sdf_feature = sdf_b.unsqueeze(-1)

    trunk_b = torch.cat([coords_batch, sdf_feature], dim=-1)

    return trunk_b



def get_cpu_memory_mb():
    if psutil is None:
        return np.nan
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return np.nan


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_loss_csv(filepath, epoch_history, loss_history):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, loss in zip(epoch_history, loss_history):
            writer.writerow([ep, loss])
    print(f"Saved: {filepath}")


def plot_loss_history(cfg, epoch_history, train_loss_history, val_loss_history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_history, train_loss_history, label="Train loss", linewidth=2)
    ax.plot(epoch_history, val_loss_history, label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE loss")
    ax.set_yscale("log")
    ax.set_title("Training and validation loss history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if cfg.save_loss_plot:
        fig.savefig(cfg.loss_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {cfg.loss_plot_path}")

    if cfg.show_loss_plot:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)


def _training_geometry_records(cfg, prepared_data, training_dataset_path):
    """Return the actual family/source composition used by this training run."""
    if str(prepared_data["mode"]).strip().lower() != "composed":
        return [
            {
                "family": str(cfg.dataset_family),
                "source_size": int(cfg.selected_dataset_size),
                "selected_count": int(prepared_data["ncases"]),
                "source_path": os.fspath(training_dataset_path),
            }
        ], None

    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(training_dataset_path)),
        "composition_manifest.json",
    )
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        records = []
        for source in manifest.get("sources", ()):
            records.append(
                {
                    "family": str(source["family"]),
                    "source_size": int(source["source_size"]),
                    "selected_count": int(source["selected_count"]),
                    "available_cases": int(
                        source.get("available_cases", source["source_size"])
                    ),
                    "source_path": os.fspath(source["source_path"]),
                }
            )
        if records:
            return records, manifest_path

    # Compatibility fallback for a composed dataset made before manifests
    # recorded source details. Family counts still come from the loaded data.
    records = []
    for family, split_counts in sorted(
        prepared_data.get("family_split_counts", {}).items()
    ):
        records.append(
            {
                "family": str(family),
                "source_size": int(
                    cfg.training_geometry_source_sizes.get(
                        family, split_counts["total"]
                    )
                ),
                "selected_count": int(split_counts["total"]),
                "source_path": "not recorded in composed dataset manifest",
            }
        )
    return records, None


def write_training_summary(
    cfg,
    device,
    model,
    ncases,
    train_count,
    val_count,
    test_count,
    epochs_completed,
    best_epoch,
    best_val_loss,
    final_test_loss,
    training_time_sec,
    cpu_memory_mb,
    gpu_peak_memory_mb,
    prepared_data,
):
    training_dataset_path = resolve_training_dataset_path(cfg)
    geometry_records, _ = _training_geometry_records(
        cfg,
        prepared_data,
        training_dataset_path,
    )
    with open(cfg.training_summary_path, "w") as f:
        f.write("DeepONet Training Summary\n")
        f.write("=" * 60 + "\n")
        f.write("Geometries and source datasets used\n")
        for record in geometry_records:
            f.write(
                f"  {record['family']}: source dataset "
                f"{record['source_size']} cases; "
                f"used {record['selected_count']} cases\n"
            )
        f.write("\n")
        f.write(f"Dataset family        : {cfg.dataset_family}\n")
        f.write(f"Dataset size          : {ncases}\n")
        f.write(f"Output folder         : {cfg.outdir}\n")
        f.write(f"Dataset mode          : {cfg.training_dataset_mode}\n")
        f.write(f"Dataset path          : {training_dataset_path}\n")
        f.write(f"Model path            : {cfg.model_path}\n")
        f.write("\n")
        f.write("Model / data\n")
        f.write("-" * 60 + "\n")
        f.write("Branch input          : full SDF field\n")
        f.write("Trunk input           : normalized x, y, SDF(x,y)\n")
        f.write(f"Latent dimension      : {cfg.latent_dim}\n")
        f.write(f"Trainable parameters  : {count_trainable_parameters(model)}\n")
        f.write(f"Number of cases       : {ncases}\n")
        f.write(f"Train / Val / Test    : {train_count} / {val_count} / {test_count}\n")
        f.write(f"Grid size             : {cfg.Nx} x {cfg.Ny}\n")
        f.write("\n")
        f.write("Training\n")
        f.write("-" * 60 + "\n")
        f.write(f"Device                : {device}\n")
        f.write(f"Epochs requested      : {cfg.epochs}\n")
        f.write(f"Epochs completed      : {epochs_completed}\n")
        f.write(f"Best epoch            : {best_epoch}\n")
        f.write(f"Best val loss         : {best_val_loss:.6e}\n")
        f.write(f"Final test loss       : {final_test_loss:.6e}\n")
        f.write(f"Learning rate         : {cfg.lr}\n")
        f.write(f"Batch size            : {cfg.batch_size}\n")
        f.write(f"Split method          : {prepared_data['split_method']}\n")
        f.write(
            f"Normalization method  : "
            f"{prepared_data['normalization_method']}\n"
        )
        if prepared_data["temperature_min"] is not None:
            f.write(
                f"Training T_min [K]    : "
                f"{prepared_data['temperature_min']:.9g}\n"
            )
            f.write(
                f"Training T_max [K]    : "
                f"{prepared_data['temperature_max']:.9g}\n"
            )
        f.write(f"Training seed         : {cfg.training_seed}\n")
        f.write(f"Deterministic mode    : {cfg.deterministic_training}\n")
        f.write(f"Training time [sec]   : {training_time_sec:.4f}\n")
        f.write("\n")
        f.write("Reproducibility files\n")
        f.write("-" * 60 + "\n")
        f.write(f"Split indices path    : {cfg.split_indices_path}\n")
        f.write(f"Config used path      : {cfg.config_used_path}\n")
        f.write(
            f"Temperature scaler    : {cfg.temperature_normalization_path}\n"
        )
        f.write("\n")
        f.write("Environment\n")
        f.write("-" * 60 + "\n")
        env_info = collect_environment_info()
        f.write(f"Python                : {env_info['python']}\n")
        f.write(f"Platform              : {env_info['platform']}\n")
        f.write(f"NumPy                 : {env_info['numpy']}\n")
        f.write(f"PyTorch               : {env_info['torch']}\n")
        f.write(f"PyTorch CUDA          : {env_info['torch_cuda']}\n")
        f.write(f"cuDNN                 : {env_info['torch_cudnn']}\n")
        f.write(f"CUDA available        : {env_info['cuda_available']}\n")
        f.write(f"CUDA device count     : {env_info['cuda_device_count']}\n")
        f.write(f"CUDA device name      : {env_info['cuda_device_name']}\n")
        f.write(f"CUBLAS config         : {env_info['cublas_workspace_config']}\n")
        f.write(f"NGSolve               : {env_info['ngsolve']}\n")
        f.write(f"Netgen                : {env_info['netgen']}\n")
        f.write("\n")
        f.write("Memory\n")
        f.write("-" * 60 + "\n")
        f.write(f"CPU memory used [MB]  : {cpu_memory_mb:.4f}\n")
        f.write(f"GPU peak memory [MB]  : {gpu_peak_memory_mb:.4f}\n")

    print(f"Saved: {cfg.training_summary_path}")


# ============================================================
# MAIN TRAINING
# ============================================================

def train_model(cfg):
    # Freeze the composition ID before loading float32 domain metadata. It is
    # the dataset identity, not a hash to recompute from rounded model settings.
    if cfg.training_dataset_mode == "composed" and not cfg.training_composition_id:
        from src.data.compose import get_composition_id
        cfg.training_composition_id = get_composition_id(cfg)
    set_deterministic_training(
        seed=cfg.training_seed,
        deterministic=cfg.deterministic_training,
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    device = get_device(cfg)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    prepared_data = load_and_prepare_training_dataset(cfg)
    sdf_all = prepared_data["sdf"]
    mask_all = prepared_data["mask"]
    temp_all = prepared_data["target_normalized"]
    coords = prepared_data["coords_normalized"]
    cfg.Lx = prepared_data["Lx"]
    cfg.Ly = prepared_data["Ly"]
    cfg.Nx = prepared_data["Nx"]
    cfg.Ny = prepared_data["Ny"]

    validate_model_input_contract(sdf_all, coords, cfg.Nx, cfg.Ny)

    ncases = sdf_all.shape[0]

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.training_seed)

    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = make_loaders(
        sdf_all,
        mask_all,
        temp_all,
        cfg,
        torch_generator=loader_generator,
        split_indices=(
            prepared_data["train_idx"],
            prepared_data["val_idx"],
            prepared_data["test_idx"],
        ),
    )

    save_split_indices(
        cfg,
        train_idx,
        val_idx,
        test_idx,
        prepared_data["split_method"],
        prepared_data["family_split_counts"],
    )
    save_temperature_normalization(cfg, prepared_data)
    save_config_snapshot(cfg)

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    model = SimpleDeepONet(
        branch_in=sdf_all.shape[1],
        trunk_in=3, #shape ius changed to three since SDF is added as a trunk feature
        latent=cfg.latent_dim
    ).to(device)

    print(f"Model is on: {next(model.parameters()).device}")
    print(f"Number of trainable parameters = {count_trainable_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    best_val = 1e30
    best_epoch = -1
    epochs_without_improvement = 0

    epoch_history = []
    train_loss_history = []
    val_loss_history = []

    for ep in range(1, cfg.epochs + 1):
        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------
        model.train()
        train_loss = 0.0

        for sdf_b, mask_b, temp_b in train_loader:
            sdf_b = sdf_b.to(device)
            mask_b = mask_b.to(device)
            temp_b = temp_b.to(device)

            trunk_b = build_case_dependent_trunk_input(coords_t, sdf_b)
            pred = model(sdf_b, trunk_b)
            loss = masked_mse(pred, temp_b, mask_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sdf_b, mask_b, temp_b in val_loader:
                sdf_b = sdf_b.to(device)
                mask_b = mask_b.to(device)
                temp_b = temp_b.to(device)

                trunk_b = build_case_dependent_trunk_input(coords_t, sdf_b)
                pred = model(sdf_b, trunk_b)
                loss = masked_mse(pred, temp_b, mask_b)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_history.append(ep)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            epochs_without_improvement = 0
            torch.save(model.state_dict(), cfg.model_path)
        else:
            epochs_without_improvement += 1

        if ep == 1 or ep % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {ep:5d} | "
                f"Train = {train_loss:.6e} | "
                f"Val = {val_loss:.6e} | "
                f"LR = {current_lr:.3e}"
            )

        if epochs_without_improvement >= cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {ep}")
            print(f"Best validation loss = {best_val:.6e}")
            break

    training_time_sec = time.time() - start_time

    # ------------------------------------------------------------
    # SAVE LOSS CSV + PLOT
    # ------------------------------------------------------------
    save_loss_csv(cfg.train_loss_csv_path, epoch_history, train_loss_history)
    save_loss_csv(cfg.val_loss_csv_path, epoch_history, val_loss_history)
    plot_loss_history(cfg, epoch_history, train_loss_history, val_loss_history)

    # ------------------------------------------------------------
    # LOAD BEST MODEL AND COMPUTE FINAL TEST LOSS
    # ------------------------------------------------------------
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

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

    cpu_memory_mb = get_cpu_memory_mb()
    gpu_peak_memory_mb = get_gpu_peak_memory_mb()

    write_training_summary(
        cfg=cfg,
        device=device,
        model=model,
        ncases=ncases,
        train_count=len(train_idx),
        val_count=len(val_idx),
        test_count=len(test_idx),
        epochs_completed=len(epoch_history),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        final_test_loss=test_loss,
        training_time_sec=training_time_sec,
        cpu_memory_mb=cpu_memory_mb,
        gpu_peak_memory_mb=gpu_peak_memory_mb,
        prepared_data=prepared_data,
    )

    print(f"\nSaved best model to: {cfg.model_path}")

    return {
        "model_path": cfg.model_path,
        "best_val_loss": best_val,
        "final_test_loss": test_loss,
        "training_time_sec": training_time_sec,
        "cpu_memory_mb": cpu_memory_mb,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "split_method": prepared_data["split_method"],
        "normalization_method": prepared_data["normalization_method"],
        "temperature_min": prepared_data["temperature_min"],
        "temperature_max": prepared_data["temperature_max"],
    }


if __name__ == "__main__":
    train_model(Settings())
