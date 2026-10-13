"""
PyTorch dataset and train/validation/test split utilities.

This baseline version uses:
    branch input = SDF only
    trunk input  = (x_normalized, y_normalized, local SDF)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DeepONetDataset(Dataset):
    def __init__(self, sdf_cases, mask_cases, temp_cases):
        self.sdf_cases = torch.tensor(sdf_cases, dtype=torch.float32)
        self.mask_cases = torch.tensor(mask_cases, dtype=torch.float32)
        self.temp_cases = torch.tensor(temp_cases, dtype=torch.float32)

    def __len__(self):
        return self.sdf_cases.shape[0]

    def __getitem__(self, idx):
        return self.sdf_cases[idx], self.mask_cases[idx], self.temp_cases[idx]


def make_split_indices(ncases, train_fraction, val_fraction, split_seed):
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(ncases)

    n_train = int(train_fraction * ncases)
    n_val = int(val_fraction * ncases)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


def _validate_explicit_split_indices(split_indices, ncases):
    train_idx, val_idx, test_idx = (
        np.asarray(values, dtype=np.int64) for values in split_indices
    )
    combined = np.concatenate((train_idx, val_idx, test_idx))
    if len(combined) != int(ncases):
        raise ValueError("Explicit split indices do not cover every case.")
    if len(np.unique(combined)) != int(ncases):
        raise ValueError("Explicit split indices overlap or contain duplicates.")
    if combined.min() < 0 or combined.max() >= int(ncases):
        raise ValueError("Explicit split indices are outside the dataset.")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    return train_idx, val_idx, test_idx


def make_loaders(
    sdf_all,
    mask_all,
    temp_all,
    cfg,
    torch_generator=None,
    split_indices=None,
):
    if split_indices is None:
        train_idx, val_idx, test_idx = make_split_indices(
            ncases=sdf_all.shape[0],
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            split_seed=cfg.split_seed,
        )
    else:
        train_idx, val_idx, test_idx = _validate_explicit_split_indices(
            split_indices,
            sdf_all.shape[0],
        )

    train_ds = DeepONetDataset(
        sdf_all[train_idx],
        mask_all[train_idx],
        temp_all[train_idx]
    )

    val_ds = DeepONetDataset(
        sdf_all[val_idx],
        mask_all[val_idx],
        temp_all[val_idx]
    )

    test_ds = DeepONetDataset(
        sdf_all[test_idx],
        mask_all[test_idx],
        temp_all[test_idx]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx
