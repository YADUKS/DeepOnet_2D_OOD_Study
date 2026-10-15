
# =============================================================================
# DATASET: dataset.npz -> PYTORCH BATCHES
#
# dataset.npz (written by data/protocol.py) holds, for every case i:
#     sdf[i]  : scaled SDF on the grid, flattened to Ny*Nx values
#     mask[i] : 1 = solid point, 0 = cavity point
#     temp[i] : scaled temperature (0 in the cavity)
# and the frozen split train_idx / val_idx / test_idx.
# With cfg.train_size = N the first N training cases PER FAMILY are used.
#
# A batch is the tuple (sdf, mask, temp), each of shape (batch, Ny*Nx).
# DeepONet branch/trunk inputs are constructed in training/runtime.py.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   data        : dictionary of all arrays stored in dataset.npz
#   ids         : case indices of one split
#   batch_size  : cases per batch (cfg.batch_size)
#   generator   : random generator that fixes the shuffling order of the
#                 training batches (seeded with cfg.training_seed)
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CASES OF ONE SPLIT, HELD AS FLOAT32 TENSORS
# =============================================================================
class FieldDataset(Dataset):
    def __init__(self, sdf_cases, mask_cases, temp_cases):
        self.sdf_cases = torch.as_tensor(sdf_cases, dtype=torch.float32)
        self.mask_cases = torch.as_tensor(mask_cases, dtype=torch.float32)
        self.temp_cases = torch.as_tensor(temp_cases, dtype=torch.float32)

    def __len__(self):
        return len(self.sdf_cases)

    def __getitem__(self, i):
        return self.sdf_cases[i], self.mask_cases[i], self.temp_cases[i]


# =============================================================================
# READ dataset.npz AND CHECK IT
#
# Refuses old (pre-protocol-2) files and splits that overlap or miss cases.
# The grid point counts are copied from the file into cfg. Physical coordinates
# are read from the data/checkpoint without rounding the generation configuration.
# =============================================================================
def training_prefix(data, per_family):
    """Balanced, nested prefixes; N ALWAYS means cases per family."""
    ids = np.asarray(data["pool_train_idx"], dtype=np.int64)
    families = np.asarray(data["geometry_family"])
    selected = []
    for family in sorted(set(families[ids].tolist())):
        family_ids = ids[families[ids] == family]
        count = len(family_ids) if per_family == 0 else int(per_family)
        if count < 1 or count > len(family_ids):
            raise ValueError(f"Training size {count} exceeds {family}'s {len(family_ids)}-case pool.")
        selected.append(family_ids[:count])
    # Interleave the families, retaining the order within every family's prefix.
    return np.array([part[k] for k in range(max(map(len, selected)))
                     for part in selected if k < len(part)], dtype=np.int64)


def load_bundle(cfg):
    from .protocol import load_and_prepare_training_dataset
    from .design import assert_unique_geometries
    # Freeze identity BEFORE reading float32 domain/grid values from disk.
    from .compose import get_composition_id
    if not cfg.training_composition_id:
        cfg.training_composition_id = get_composition_id(cfg)
    prepared = load_and_prepare_training_dataset(cfg)
    with np.load(cfg.dataset_path, allow_pickle=True) as saved:
        data = dict(saved)
    assert_unique_geometries(data["geometry_keys"], data["roles"], data["metadata_json"])
    data["temp"] = prepared["target_normalized"]
    data["T_min"] = np.array([prepared["temperature_min"]])
    data["T_max"] = np.array([prepared["temperature_max"]])
    data["pool_train_idx"] = prepared["train_idx"]
    data["train_idx"] = training_prefix(data, cfg.train_size)
    data["family_counts"] = {str(f): int(sum(data["geometry_family"][data["train_idx"]] == f))
                             for f in sorted(set(data["geometry_family"].tolist()))}
    for key in ("Nx", "Ny"):
        setattr(cfg, key, data[key][0].item())
    return data


# =============================================================================
# ONE DATALOADER PER SPLIT
#
# Only the training loader shuffles; its order is fixed by the seeded
# generator, so a run with the same seed sees the same batches.
# Returns: train_loader, val_loader, test_loader
# =============================================================================
def make_loaders(data, cfg):
    generator = torch.Generator().manual_seed(cfg.training_seed)
    pin = cfg.device == "cuda"

    loaders = []
    for split in ("train_idx", "val_idx", "test_idx"):
        ids = data[split]
        dataset = FieldDataset(data["sdf"][ids], data["mask"][ids], data["temp"][ids])

        if split == "train_idx":
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                                generator=generator, pin_memory=pin, num_workers=0)
        else:
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=pin, num_workers=0)
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
