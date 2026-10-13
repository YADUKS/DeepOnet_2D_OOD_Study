"""
Resolve and validate datasets before they reach DeepONet training.

The fixed model contract is:
    branch = complete flattened SDF field
    trunk  = normalized x, normalized y, and local SDF(x,y)
"""

import os

import numpy as np
from src.data.design import file_digest as sha256_file, assert_unique_geometries

from src.data.compose import get_composition_id
from src.fem.geometry_spec import canonical_geometry_name, TRAINING_GEOMETRY_FAMILIES
from src.data.design import DESIGN, role_splits


def resolve_training_dataset_path(cfg):
    """Resolve the configured legacy or composed training dataset."""
    mode = str(cfg.training_dataset_mode).strip().lower()
    if mode == "legacy":
        return cfg.dataset_path
    if mode == "composed":
        composition_id = str(cfg.training_composition_id).strip()
        if not composition_id:
            composition_id = get_composition_id(cfg)
        return cfg.composed_dataset_path(composition_id)
    raise ValueError(
        "training_dataset_mode must be either 'legacy' or 'composed'."
    )


def _single_value(data, name, cast):
    if name not in data:
        raise ValueError(f"Training dataset is missing required field {name!r}.")
    values = np.asarray(data[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Training field {name!r} must contain one value.")
    return cast(values[0])


def _require_finite(name, array):
    if not np.isfinite(array).all():
        invalid = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name} contains {invalid} non-finite values.")


def validate_training_dataset(data, dataset_path, mode):
    """
    Validate geometry arrays and return resolved dataset information.

    Temperature requirements differ by mode: legacy files contain normalized
    ``temp``; composed files preserve ``temp_physical`` for Change 6.
    """
    mode = str(mode).strip().lower()
    common_required = ("sdf", "mask", "coords", "Lx", "Ly", "Nx", "Ny")
    missing = [name for name in common_required if name not in data]
    if missing:
        raise ValueError(f"Training dataset is missing fields: {missing}")

    sdf = np.asarray(data["sdf"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32)
    coords = np.asarray(data["coords"], dtype=np.float32)
    Lx = _single_value(data, "Lx", float)
    Ly = _single_value(data, "Ly", float)
    Nx = _single_value(data, "Nx", int)
    Ny = _single_value(data, "Ny", int)

    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError("Domain dimensions Lx and Ly must be positive.")
    if Nx < 2 or Ny < 2:
        raise ValueError("Cartesian grid dimensions Nx and Ny must be at least 2.")
    if sdf.ndim != 2:
        raise ValueError(f"SDF must have shape [cases, points], got {sdf.shape}.")
    if len(sdf) == 0:
        raise ValueError("Training dataset contains no cases.")
    if mask.shape != sdf.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match SDF {sdf.shape}.")

    expected_points = Nx * Ny
    if sdf.shape[1] != expected_points:
        raise ValueError(
            f"Branch width {sdf.shape[1]} does not match Nx*Ny={expected_points}."
        )
    if coords.shape != (expected_points, 2):
        raise ValueError(
            f"Coordinates must have shape {(expected_points, 2)}, got {coords.shape}."
        )

    _require_finite("SDF", sdf)
    _require_finite("mask", mask)
    _require_finite("coordinates", coords)
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("Mask must contain binary values 0 and 1 only.")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("At least one case contains no solid grid points.")
    if np.any(mask.sum(axis=1) == mask.shape[1]):
        raise ValueError("At least one case contains no hollow grid points.")

    sign_mask = (sdf >= 0.0).astype(np.float32)
    sign_disagreement = np.count_nonzero(sign_mask != mask)
    if sign_disagreement:
        raise ValueError(
            f"Mask disagrees with the SDF sign at {sign_disagreement} grid values."
        )

    tolerance = 1e-6
    if coords[:, 0].min() < -tolerance or coords[:, 0].max() > Lx + tolerance:
        raise ValueError("Physical x coordinates lie outside [0, Lx].")
    if coords[:, 1].min() < -tolerance or coords[:, 1].max() > Ly + tolerance:
        raise ValueError("Physical y coordinates lie outside [0, Ly].")
    if not np.isclose(coords[:, 0].min(), 0.0, atol=tolerance):
        raise ValueError("Cartesian x coordinates do not start at zero.")
    if not np.isclose(coords[:, 0].max(), Lx, atol=tolerance):
        raise ValueError("Cartesian x coordinates do not end at Lx.")
    if not np.isclose(coords[:, 1].min(), 0.0, atol=tolerance):
        raise ValueError("Cartesian y coordinates do not start at zero.")
    if not np.isclose(coords[:, 1].max(), Ly, atol=tolerance):
        raise ValueError("Cartesian y coordinates do not end at Ly.")

    normalized_coords = coords.copy()
    normalized_coords[:, 0] /= Lx
    normalized_coords[:, 1] /= Ly
    if (
        normalized_coords.min() < -tolerance
        or normalized_coords.max() > 1.0 + tolerance
    ):
        raise ValueError("Normalized coordinates lie outside [0, 1].")

    if mode == "legacy":
        if "temp" not in data:
            raise ValueError("Legacy training dataset is missing normalized 'temp'.")
        target = np.asarray(data["temp"], dtype=np.float32)
        if target.shape != sdf.shape:
            raise ValueError(
                f"Temperature shape {target.shape} does not match SDF {sdf.shape}."
            )
        _require_finite("normalized temperature", target)
        target_kind = "normalized"
    elif mode == "composed":
        if "temp_physical" not in data:
            raise ValueError(
                "Composed training pool is missing physical temperature data."
            )
        target = np.asarray(data["temp_physical"], dtype=np.float32)
        if target.shape != sdf.shape:
            raise ValueError(
                f"Physical temperature shape {target.shape} does not match "
                f"SDF {sdf.shape}."
            )
        _require_finite("physical temperature", target)
        if "geometry_family" not in data:
            raise ValueError(
                "Composed training pool is missing per-case geometry families."
            )
        geometry_family = np.asarray(data["geometry_family"], dtype=object)
        if geometry_family.shape != (len(sdf),):
            raise ValueError(
                "Composed geometry_family must contain one label per case."
            )
        geometry_family = np.asarray(
            [canonical_geometry_name(value) for value in geometry_family],
            dtype=object,
        )
        target_kind = "physical_pending_training_normalization"
    else:
        raise ValueError("Validation mode must be 'legacy' or 'composed'.")

    result = {
        "dataset_path": os.fspath(dataset_path),
        "mode": mode,
        "sdf": sdf,
        "mask": mask,
        "target": target,
        "target_kind": target_kind,
        "coords": coords,
        "coords_normalized": normalized_coords.astype(np.float32),
        "Lx": Lx,
        "Ly": Ly,
        "Nx": Nx,
        "Ny": Ny,
        "ncases": int(sdf.shape[0]),
        "npoints": int(expected_points),
        "branch_features": int(expected_points),
        "trunk_features": 3,
    }
    if mode == "composed":
        result["geometry_family"] = geometry_family
        if not set(geometry_family.tolist()).issubset(TRAINING_GEOMETRY_FAMILIES):
            raise ValueError("Composed data contains an unsupported or OOD family.")
        if "design" not in data or str(np.asarray(data["design"])[0]) != DESIGN:
            raise ValueError("Composed dataset must use the v2 Halton/LHS protocol; regenerate and compose.")
        result["roles"] = np.asarray(data["roles"])
        assert_unique_geometries(data["geometry_keys"], result["roles"], data["metadata_json"])
        splits = role_splits(result["roles"], geometry_family)
        for name, expected in zip(("train_idx", "val_idx", "test_idx"), splits[:3]):
            if name not in data or not np.array_equal(data[name], expected):
                raise ValueError(f"Saved {name} disagrees with frozen dataset roles.")
    return result


def load_and_validate_training_dataset(cfg):
    """Load, validate, and return the configured training data contract."""
    dataset_path = resolve_training_dataset_path(cfg)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    try:
        validated = validate_training_dataset(
            data,
            dataset_path,
            cfg.training_dataset_mode,
        )
    finally:
        data.close()
    return validated


def validate_model_input_contract(sdf, normalized_coords, Nx, Ny):
    """Validate the branch/trunk dimensions immediately before model creation."""
    sdf = np.asarray(sdf)
    normalized_coords = np.asarray(normalized_coords)
    expected_points = int(Nx) * int(Ny)

    if sdf.ndim != 2 or sdf.shape[1] != expected_points:
        raise ValueError(
            "Branch input must be the complete flattened SDF field with "
            f"{expected_points} values per case; got {sdf.shape}."
        )
    if normalized_coords.shape != (expected_points, 2):
        raise ValueError(
            "Normalized coordinate input must contain x and y for every "
            f"Cartesian point; got {normalized_coords.shape}."
        )

    trunk_feature_count = normalized_coords.shape[1] + 1
    if trunk_feature_count != 3:
        raise ValueError(
            "Trunk input must be exactly [x_normalized, y_normalized, sdf(x,y)]."
        )
    return {
        "branch_features": expected_points,
        "trunk_features": trunk_feature_count,
    }


def _validate_split_fractions(train_fraction, val_fraction):
    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    test_fraction = 1.0 - train_fraction - val_fraction
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")
    if test_fraction <= 0.0:
        raise ValueError("train_fraction + val_fraction must be below 1.")
    return train_fraction, val_fraction, test_fraction


def _allocate_family_split_counts(ncases, train_fraction, val_fraction):
    if int(ncases) < 3:
        raise ValueError(
            "Each selected geometry family requires at least three cases "
            "for stratified train/validation/test splitting."
        )
    ncases = int(ncases)
    n_train = max(1, int(np.floor(train_fraction * ncases)))
    n_val = max(1, int(np.floor(val_fraction * ncases)))
    if n_train + n_val >= ncases:
        excess = n_train + n_val - (ncases - 1)
        if n_train >= n_val and n_train - excess >= 1:
            n_train -= excess
        else:
            n_val -= excess
    n_test = ncases - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Could not allocate non-empty family-level splits.")
    return n_train, n_val, n_test


def make_stratified_split_indices(
    geometry_family,
    train_fraction,
    val_fraction,
    split_seed,
):
    """Create deterministic train/validation/test splits within every family."""
    train_fraction, val_fraction, _ = _validate_split_fractions(
        train_fraction,
        val_fraction,
    )
    geometry_family = np.asarray(geometry_family, dtype=object)
    if geometry_family.ndim != 1 or len(geometry_family) == 0:
        raise ValueError("geometry_family must be a non-empty one-dimensional array.")

    train_parts = []
    val_parts = []
    test_parts = []
    family_counts = {}

    for family_number, family in enumerate(sorted(set(geometry_family.tolist()))):
        family_indices = np.flatnonzero(geometry_family == family)
        n_train, n_val, n_test = _allocate_family_split_counts(
            len(family_indices),
            train_fraction,
            val_fraction,
        )
        family_seed = int(split_seed) + 1009 * (family_number + 1)
        rng = np.random.default_rng(family_seed)
        shuffled = family_indices[rng.permutation(len(family_indices))]
        train_parts.append(shuffled[:n_train])
        val_parts.append(shuffled[n_train:n_train + n_val])
        test_parts.append(shuffled[n_train + n_val:])
        family_counts[str(family)] = {
            "total": int(len(family_indices)),
            "train": int(n_train),
            "validation": int(n_val),
            "test": int(n_test),
        }

    shuffle_rng = np.random.default_rng(int(split_seed) + 99991)
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    train_idx = train_idx[shuffle_rng.permutation(len(train_idx))]
    val_idx = val_idx[shuffle_rng.permutation(len(val_idx))]
    test_idx = test_idx[shuffle_rng.permutation(len(test_idx))]
    return train_idx, val_idx, test_idx, family_counts


def make_random_split_indices(ncases, train_fraction, val_fraction, split_seed):
    """Retain the legacy deterministic whole-dataset split."""
    _validate_split_fractions(train_fraction, val_fraction)
    rng = np.random.default_rng(int(split_seed))
    indices = rng.permutation(int(ncases))
    n_train = int(float(train_fraction) * int(ncases))
    n_val = int(float(val_fraction) * int(ncases))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Legacy train/validation/test split is empty.")
    return train_idx, val_idx, test_idx


def fit_training_temperature_normalization(
    physical_temperature,
    mask,
    train_idx,
):
    """Fit min/max using solid grid values from training cases only."""
    physical_temperature = np.asarray(physical_temperature, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    training_values = physical_temperature[train_idx][mask[train_idx] > 0.5]
    training_values = training_values[np.isfinite(training_values)]
    if training_values.size == 0:
        raise ValueError("Training split contains no finite solid temperatures.")
    temperature_min = float(training_values.min())
    temperature_max = float(training_values.max())
    if not temperature_max > temperature_min:
        raise ValueError("Training temperature range must be strictly positive.")
    return temperature_min, temperature_max


def apply_temperature_normalization(
    physical_temperature,
    mask,
    temperature_min,
    temperature_max,
):
    """Apply fixed training statistics to every case and zero the hollow region."""
    scale = float(temperature_max) - float(temperature_min)
    if scale <= 0.0:
        raise ValueError("Temperature normalization scale must be positive.")
    normalized = (
        np.asarray(physical_temperature, dtype=np.float32)
        - float(temperature_min)
    ) / scale
    normalized *= np.asarray(mask, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized temperature contains non-finite values.")
    return normalized.astype(np.float32)


def load_and_prepare_training_dataset(cfg):
    """
    Load data, create reproducible splits, and normalize without data leakage.
    """
    validated = load_and_validate_training_dataset(cfg)
    mode = validated["mode"]

    if mode == "composed":
        train_idx, val_idx, test_idx, family_counts = (
            role_splits(validated["roles"], validated["geometry_family"])
        )
        temperature_min, temperature_max = (
            fit_training_temperature_normalization(
                validated["target"],
                validated["mask"],
                train_idx,
            )
        )
        normalized_target = apply_temperature_normalization(
            validated["target"],
            validated["mask"],
            temperature_min,
            temperature_max,
        )
        split_method = "frozen_halton_train_lhs_holdouts_v2"
        normalization_method = "training_solid_minmax"
    else:
        train_idx, val_idx, test_idx = make_random_split_indices(
            validated["ncases"],
            cfg.train_fraction,
            cfg.val_fraction,
            cfg.split_seed,
        )
        normalized_target = validated["target"]
        temperature_min = None
        temperature_max = None
        family_counts = {}
        split_method = "legacy_deterministic_random"
        normalization_method = "pre_normalized_legacy_dataset"

    return {
        **validated,
        "target_normalized": normalized_target,
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "split_method": split_method,
        "family_split_counts": family_counts,
        "normalization_method": normalization_method,
        "temperature_min": temperature_min,
        "temperature_max": temperature_max,
    }
