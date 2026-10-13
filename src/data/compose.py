"""
Compose selected reusable geometry-family files into one training data pool.

This stage preserves the fixed train/validation/test roles and does not fit
temperature normalization.  It preserves physical temperatures so those operations can be
performed using the training partition only.
"""

import hashlib
import json
import os
from dataclasses import asdict

import numpy as np

from src.fem.geometry_spec import TRAINING_GEOMETRY_FAMILIES, canonical_geometry_name


from src.data.design import DESIGN, design_id, relative_path, resolve_reference, role_splits, assert_unique_geometries, file_digest

COMPOSED_SCHEMA_VERSION = 2


def _scalar(data, name):
    if name not in data:
        raise ValueError(f"Source dataset is missing required field {name!r}.")
    values = np.asarray(data[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Source field {name!r} must contain one value.")
    return values[0]


def _composition_spec(cfg):
    selected = []
    for requested_family, requested_count in cfg.training_geometry_counts.items():
        family = canonical_geometry_name(requested_family)
        if family not in TRAINING_GEOMETRY_FAMILIES:
            raise ValueError(f"Unsupported training family: {family!r}")
        count = int(requested_count)
        if count < 0:
            raise ValueError(f"Training count for {family} cannot be negative.")
        if count == 0:
            continue
        if family not in cfg.training_geometry_source_sizes:
            raise KeyError(f"No source dataset size configured for {family!r}.")
        source_size = int(cfg.training_geometry_source_sizes[family])
        if source_size <= 0:
            raise ValueError(f"Source dataset size for {family} must be positive.")
        selected.append(
            {
                "family": family,
                "source_size": source_size,
                "selected_count": count,
                "design_id": design_id(cfg, family),
                "source_path": cfg.geometry_dataset_path(family, source_size),
            }
        )

    if not selected:
        raise ValueError("At least one training geometry count must be positive.")
    selected.sort(key=lambda item: item["family"])
    return {
        "composition_seed": int(cfg.composition_seed),
        "selected_families": selected,
    }


def get_composition_id(cfg):
    """Return a stable ID determined by selected sources, counts, and seed."""
    spec = _composition_spec(cfg)
    spec["design"] = DESIGN
    for source in spec["selected_families"]:
        source.pop("source_path")  # identity must survive repository relocation
    encoded = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _validate_source_dataset(data, source, reference):
    required_arrays = (
        "sdf",
        "mask",
        "temp_physical",
        "metadata_json",
        "fe_paths",
        "coords",
        "Lx",
        "Ly",
        "Nx",
        "Ny",
        "maxh",
        "order",
        "kappa",
        "h_hole",
        "T_inf",
        "q_top",
        "geometry_family",
        "n_cases",
    )
    missing = [name for name in required_arrays if name not in data]
    if missing:
        raise ValueError(
            f"Source dataset {source['source_path']} is missing: {missing}"
        )

    stored_family = canonical_geometry_name(str(_scalar(data, "geometry_family")))
    if stored_family != source["family"]:
        raise ValueError(
            f"Expected family {source['family']!r}, found {stored_family!r} "
            f"in {source['source_path']}."
        )

    if str(_scalar(data, "design")) != DESIGN or int(_scalar(data, "schema_version")) != 2:
        raise ValueError("Regenerate this source using the Halton/LHS v2 protocol.")
    if str(_scalar(data, "geometry_group")) != "training" or not bool(_scalar(data, "training_eligible")):
        raise ValueError("OOD datasets cannot enter a training composition.")
    roles = np.asarray(data["roles"])
    assert_unique_geometries(data["geometry_keys"], roles, data["metadata_json"])
    role_splits(roles, np.full(len(roles), stored_family))
    if source["selected_count"] > np.count_nonzero(roles == "train"):
        raise ValueError("Selected count exceeds the available Halton training pool.")
    ncases = int(_scalar(data, "n_cases"))
    case_array_lengths = {
        "sdf": len(data["sdf"]),
        "mask": len(data["mask"]),
        "temp_physical": len(data["temp_physical"]),
        "metadata_json": len(data["metadata_json"]),
        "fe_paths": len(data["fe_paths"]),
    }
    if any(length != ncases for length in case_array_lengths.values()):
        raise ValueError(
            f"Inconsistent case-array lengths in {source['source_path']}: "
            f"{case_array_lengths}, declared n_cases={ncases}."
        )
    if source["selected_count"] > ncases:
        raise ValueError(
            f"Requested {source['selected_count']} {source['family']} cases "
            f"but {source['source_path']} contains only {ncases}."
        )

    current = {
        "Lx": float(_scalar(data, "Lx")),
        "Ly": float(_scalar(data, "Ly")),
        "Nx": int(_scalar(data, "Nx")),
        "Ny": int(_scalar(data, "Ny")),
        "maxh": float(_scalar(data, "maxh")),
        "order": int(_scalar(data, "order")),
        "kappa": float(_scalar(data, "kappa")),
        "h_hole": float(_scalar(data, "h_hole")),
        "T_inf": float(_scalar(data, "T_inf")),
        "q_top": float(_scalar(data, "q_top")),
        "coords": np.asarray(data["coords"], dtype=np.float32),
        "npoints": int(data["sdf"].shape[1]),
    }
    if data["sdf"].ndim != 2:
        raise ValueError(f"SDF in {source['source_path']} must be two-dimensional.")
    expected_points = current["Nx"] * current["Ny"]
    if current["npoints"] != expected_points:
        raise ValueError(
            f"SDF point count {current['npoints']} does not equal "
            f"Nx*Ny={expected_points} in {source['source_path']}."
        )
    if data["mask"].shape != data["sdf"].shape:
        raise ValueError(f"Mask and SDF shapes differ in {source['source_path']}.")
    if data["temp_physical"].shape != data["sdf"].shape:
        raise ValueError(
            f"Physical temperature and SDF shapes differ in {source['source_path']}."
        )
    if current["coords"].shape != (expected_points, 2):
        raise ValueError(
            f"Coordinate shape is invalid in {source['source_path']}: "
            f"{current['coords'].shape}."
        )

    if reference is not None:
        for name in ("Lx", "Ly", "maxh", "kappa", "h_hole", "T_inf", "q_top"):
            if not np.isclose(current[name], reference[name], rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"Incompatible {name} in {source['source_path']}: "
                    f"{current[name]} != {reference[name]}."
                )
        for name in ("Nx", "Ny", "npoints", "order"):
            if current[name] != reference[name]:
                raise ValueError(
                    f"Incompatible {name} in {source['source_path']}: "
                    f"{current[name]} != {reference[name]}."
                )
        if not np.array_equal(current["coords"], reference["coords"]):
            raise ValueError(
                f"Grid coordinates differ in {source['source_path']}."
            )
    return current


def _family_selection_indices(roles, selected):
    """Keep the ordered training prefix and every fixed holdout case."""
    roles = np.asarray(roles)
    return np.concatenate((np.flatnonzero(roles == "train")[:selected],
                           np.flatnonzero(roles == "val"), np.flatnonzero(roles == "test")))


def compose_training_geometry_dataset(cfg):
    """
    Select Halton prefixes and preserve fixed LHS holdouts across families.
    """
    specification = _composition_spec(cfg)
    composition_id = get_composition_id(cfg)
    output_path = cfg.composed_dataset_path(composition_id)
    output_dir = os.path.dirname(output_path)
    if os.path.isfile(output_path) and not cfg.force_generation:
        with np.load(output_path, allow_pickle=True) as existing:
            if str(_scalar(existing, "design")) != DESIGN or str(_scalar(existing, "composition_id")) != composition_id:
                raise ValueError("Existing composition has incompatible provenance; use a new configuration.")
            role_splits(existing["roles"], existing["geometry_family"])
            assert_unique_geometries(existing["geometry_keys"], existing["roles"], existing["metadata_json"])
            with open(os.path.join(output_dir, "composition_manifest.json"), encoding="utf-8") as stream:
                manifest = json.load(stream)
            for source in manifest["sources"]:
                original = resolve_reference(output_path, source["relative_source_path"])
                if file_digest(original) != source["source_sha256"]:
                    raise ValueError("Source dataset changed after composition. Recompose with --force.")
        print(f"Reusing completed composition: {output_path}")
        return output_path
    os.makedirs(output_dir, exist_ok=True)

    sdf_parts = []
    mask_parts = []
    temperature_parts = []
    metadata_parts = []
    fe_path_parts = []
    family_parts = []
    source_path_parts = []
    source_case_id_parts = []
    source_candidate_id_parts = []
    role_parts = []
    geometry_key_parts = []
    reference = None
    source_records = []

    for source in specification["selected_families"]:
        source_path = source["source_path"]
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Required {source['family']} dataset does not exist: {source_path}"
            )

        with np.load(source_path, allow_pickle=True) as data:
            current = _validate_source_dataset(data, source, reference)
            if reference is None:
                reference = current

            available = int(_scalar(data, "n_cases"))
            indices = _family_selection_indices(data["roles"], source["selected_count"])
            role_parts.append(np.asarray(data["roles"])[indices])
            geometry_key_parts.append(np.asarray(data["geometry_keys"])[indices])
            sdf_parts.append(np.asarray(data["sdf"][indices], dtype=np.float32))
            mask_parts.append(np.asarray(data["mask"][indices], dtype=np.float32))
            temperature_parts.append(
                np.asarray(data["temp_physical"][indices], dtype=np.float32)
            )
            metadata_parts.append(
                np.asarray(data["metadata_json"][indices], dtype=object)
            )
            fe_path_parts.append(
                np.asarray([relative_path(resolve_reference(source_path, p), output_path) for p in data["fe_paths"][indices]], dtype=str)
            )
            family_parts.append(
                np.full(len(indices), source["family"], dtype=object)
            )
            source_path_parts.append(
                np.full(len(indices), relative_path(source_path, output_path), dtype=object)
            )
            if "case_ids" in data:
                source_case_id_parts.append(
                    np.asarray(data["case_ids"][indices], dtype=np.int32)
                )
            else:
                source_case_id_parts.append(indices.astype(np.int32))
            if "candidate_ids" in data:
                source_candidate_id_parts.append(
                    np.asarray(data["candidate_ids"][indices], dtype=np.int32)
                )
            else:
                source_candidate_id_parts.append(
                    np.full(len(indices), -1, dtype=np.int32)
                )

            source_records.append(
                {
                    **source,
                    "relative_source_path": relative_path(source_path, output_path),
                    "source_sha256": file_digest(source_path),
                    "available_cases": available,
                    "selected_source_indices": indices.tolist(),
                }
            )

    sdf = np.concatenate(sdf_parts, axis=0)
    mask = np.concatenate(mask_parts, axis=0)
    temp_physical = np.concatenate(temperature_parts, axis=0)
    metadata_json = np.concatenate(metadata_parts, axis=0)
    fe_paths = np.concatenate(fe_path_parts, axis=0)
    geometry_family = np.concatenate(family_parts, axis=0)
    source_dataset_path = np.concatenate(source_path_parts, axis=0)
    source_case_id = np.concatenate(source_case_id_parts, axis=0)
    source_candidate_id = np.concatenate(source_candidate_id_parts, axis=0)

    roles = np.concatenate(role_parts)
    geometry_keys = np.concatenate(geometry_key_parts)
    overlap_check = assert_unique_geometries(geometry_keys, roles, metadata_json)
    if len(set(geometry_keys.tolist())) != len(geometry_keys):
        raise ValueError("Duplicate geometry records in composed dataset.")
    train_idx, val_idx, test_idx, family_counts = role_splits(roles, geometry_family)
    permutation = np.arange(len(sdf))  # membership/order fixed; DataLoader shuffles batches

    np.savez(
        output_path,
        schema_version=np.array([COMPOSED_SCHEMA_VERSION], dtype=np.int32),
        design=np.array([DESIGN]), roles=roles, geometry_keys=geometry_keys,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        composition_id=np.array([composition_id], dtype=object),
        geometry_group=np.array(["training_pool"], dtype=object),
        sdf=sdf,
        mask=mask,
        temp_physical=temp_physical,
        metadata_json=metadata_json,
        fe_paths=fe_paths,
        geometry_family=geometry_family,
        source_dataset_path=source_dataset_path,
        source_case_id=source_case_id,
        source_candidate_id=source_candidate_id,
        composed_case_id=np.arange(len(sdf), dtype=np.int32),
        composition_permutation=permutation.astype(np.int32),
        coords=reference["coords"],
        Lx=np.array([reference["Lx"]], dtype=np.float32),
        Ly=np.array([reference["Ly"]], dtype=np.float32),
        Nx=np.array([reference["Nx"]], dtype=np.int32),
        Ny=np.array([reference["Ny"]], dtype=np.int32),
        maxh=np.array([reference["maxh"]], dtype=np.float32),
        order=np.array([reference["order"]], dtype=np.int32),
        kappa=np.array([reference["kappa"]], dtype=np.float64),
        h_hole=np.array([reference["h_hole"]], dtype=np.float64),
        T_inf=np.array([reference["T_inf"]], dtype=np.float64),
        q_top=np.array([reference["q_top"]], dtype=np.float64),
        n_cases=np.array([len(sdf)], dtype=np.int32),
        normalization_status=np.array(
            ["physical_temperature_not_yet_normalized"],
            dtype=object,
        ),
    )

    manifest = {
        "schema_version": COMPOSED_SCHEMA_VERSION,
        "composition_id": composition_id,
        "output_path": output_path,
        "total_cases": int(len(sdf)),
        "design": DESIGN, "family_split_counts": family_counts,
        "overlap_check": overlap_check,
        "composition_seed": int(cfg.composition_seed),
        "sources": source_records,
        "combined_family_order": geometry_family.tolist(),
        "config": asdict(cfg),
        "normalization_status": "physical_temperature_not_yet_normalized",
    }
    manifest_path = os.path.join(output_dir, "composition_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print(f"Saved composed training pool: {output_path}")
    print(f"Saved composition manifest: {manifest_path}")
    return output_path
