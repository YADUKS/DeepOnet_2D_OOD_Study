"""
Evaluate the trained model on the exact held-out composed simple-geometry set.

Only indices stored in ``split_indices.npz:test_idx`` are evaluated.  Training
and validation indices are checked for disjointness before inference.
"""

import json
import os
import platform
import sys
from dataclasses import asdict

import numpy as np
from src.data.design import resolve_reference, file_digest, role_splits
import torch

from src.analysis.ood import (
    compute_case_metrics,
    evaluate_prediction_on_fe_mesh,
    get_device,
    load_training_model_settings,
    load_training_temperature_scaler,
    predict_ood_dataset,
    save_case_figure,
    save_family_comparison_figure,
    save_metrics_csv,
    save_original_style_fe_outputs,
    get_selected_case_output_dir,
    save_legacy_style_case_bundle,
    save_rmse_mape_plot_and_csv,
    summarize_metrics,
)
from src.fem.geometry_spec import (
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)
from src.models.deeponet import SimpleDeepONet
from src.data.protocol import (
    apply_temperature_normalization,
    resolve_training_dataset_path,
)


def load_exact_saved_splits(cfg, ncases):
    """Load saved splits and prove they are complete and non-overlapping."""
    path = cfg.split_indices_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Saved training split file is missing: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = ("train_idx", "val_idx", "test_idx", "split_method")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Saved split file is missing fields: {missing}")
        train_idx = np.asarray(data["train_idx"], dtype=np.int64)
        val_idx = np.asarray(data["val_idx"], dtype=np.int64)
        test_idx = np.asarray(data["test_idx"], dtype=np.int64)
        split_method = str(np.asarray(data["split_method"]).reshape(-1)[0])
        family_counts_json = (
            str(np.asarray(data["family_split_counts_json"]).reshape(-1)[0])
            if "family_split_counts_json" in data
            else "{}"
        )
    if split_method != "frozen_halton_train_lhs_holdouts_v2":
        raise ValueError(
            "Composed evaluation requires frozen_halton_train_lhs_holdouts_v2 "
            f"splits; found {split_method!r}."
        )
    combined = np.concatenate((train_idx, val_idx, test_idx))
    if len(combined) != int(ncases):
        raise ValueError("Saved splits do not cover every composed case.")
    if len(np.unique(combined)) != int(ncases):
        raise ValueError("Saved training/validation/test indices overlap.")
    if combined.min() < 0 or combined.max() >= int(ncases):
        raise ValueError("Saved split indices lie outside the composed dataset.")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Saved train/validation/test split is empty.")
    return {
        "path": path,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_method": split_method,
        "family_counts": json.loads(family_counts_json),
    }


def load_composed_test_dataset(cfg, training_settings, scaler):
    """Load the composed pool and select only its exact saved test indices."""
    path = resolve_training_dataset_path(cfg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Composed training pool does not exist: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = (
            "sdf",
            "mask",
            "temp_physical",
            "coords",
            "geometry_family",
            "fe_paths",
            "metadata_json",
            "n_cases",
            "Nx",
            "Ny",
            "Lx",
            "Ly",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Composed dataset is missing fields: {missing}")
        ncases = int(np.asarray(data["n_cases"]).reshape(-1)[0])
        sdf_all = np.asarray(data["sdf"], dtype=np.float32)
        mask_all = np.asarray(data["mask"], dtype=np.float32)
        temperature_all = np.asarray(data["temp_physical"], dtype=np.float32)
        coords = np.asarray(data["coords"], dtype=np.float32)
        families_all = np.asarray(data["geometry_family"], dtype=object)
        fe_paths_all = np.asarray([resolve_reference(path, p) for p in data["fe_paths"]], dtype=object)
        metadata_all = np.asarray(data["metadata_json"], dtype=object)
        Nx = int(np.asarray(data["Nx"]).reshape(-1)[0])
        Ny = int(np.asarray(data["Ny"]).reshape(-1)[0])
        Lx = float(np.asarray(data["Lx"]).reshape(-1)[0])
        Ly = float(np.asarray(data["Ly"]).reshape(-1)[0])
        composed_case_ids = (
            np.asarray(data["composed_case_id"], dtype=np.int32)
            if "composed_case_id" in data
            else np.arange(ncases, dtype=np.int32)
        )
        source_case_ids = (
            np.asarray(data["source_case_id"], dtype=np.int32)
            if "source_case_id" in data
            else np.full(ncases, -1, dtype=np.int32)
        )
        source_candidate_ids = (
            np.asarray(data["source_candidate_id"], dtype=np.int32)
            if "source_candidate_id" in data
            else np.full(ncases, -1, dtype=np.int32)
        )

    expected_points = Nx * Ny
    if sdf_all.shape != (ncases, expected_points):
        raise ValueError(f"Invalid composed SDF shape: {sdf_all.shape}.")
    if mask_all.shape != sdf_all.shape or temperature_all.shape != sdf_all.shape:
        raise ValueError("Composed mask/temperature shapes do not match SDF.")
    if coords.shape != (expected_points, 2):
        raise ValueError(f"Invalid composed coordinate shape: {coords.shape}.")
    if any(
        array.shape != (ncases,)
        for array in (
            families_all,
            fe_paths_all,
            metadata_all,
            composed_case_ids,
            source_case_ids,
            source_candidate_ids,
        )
    ):
        raise ValueError("Composed provenance arrays must have one value per case.")
    if not np.isfinite(sdf_all).all() or not np.isfinite(mask_all).all():
        raise ValueError("Composed SDF or mask contains non-finite values.")
    if not np.isfinite(temperature_all).all() or not np.isfinite(coords).all():
        raise ValueError("Composed temperature or coordinates are non-finite.")
    if np.count_nonzero((sdf_all >= 0.0).astype(np.float32) != mask_all):
        raise ValueError("Composed mask disagrees with the SDF sign.")
    if (
        Nx != training_settings["Nx"]
        or Ny != training_settings["Ny"]
        or not np.isclose(Lx, training_settings["Lx"])
        or not np.isclose(Ly, training_settings["Ly"])
    ):
        raise ValueError("Composed dataset grid does not match the trained model.")
    if scaler.get("training_dataset_sha256"):
        matches_training = file_digest(path) == scaler["training_dataset_sha256"]
    else:
        matches_training = os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(scaler["training_dataset_path"]))
    if not matches_training:
        raise ValueError(
            "The saved training scaler belongs to a different composed dataset."
        )

    splits = load_exact_saved_splits(cfg, ncases)
    with np.load(path, allow_pickle=True) as saved:
        expected = role_splits(saved["roles"], families_all)
    for name, indices in zip(("train_idx", "val_idx", "test_idx"), expected[:3]):
        if not np.array_equal(splits[name], indices):
            raise ValueError(f"Model {name} disagrees with the dataset's frozen roles.")
    test_idx = splits["test_idx"]
    coords_normalized = coords.copy()
    coords_normalized[:, 0] /= Lx
    coords_normalized[:, 1] /= Ly
    families = np.asarray(
        [canonical_geometry_name(value) for value in families_all[test_idx]],
        dtype=object,
    )
    return {
        "path": path,
        "family": "composed_simple_test",
        "sdf": sdf_all[test_idx],
        "mask": mask_all[test_idx],
        "temp_physical": temperature_all[test_idx],
        "coords_normalized": coords_normalized.astype(np.float32),
        "geometry_family": families,
        "fe_paths": fe_paths_all[test_idx],
        "metadata_json": metadata_all[test_idx],
        "case_ids": composed_case_ids[test_idx],
        "source_case_ids": source_case_ids[test_idx],
        "candidate_ids": source_candidate_ids[test_idx],
        "original_composed_indices": test_idx,
        "ncases": len(test_idx),
        "Nx": Nx,
        "Ny": Ny,
        "Lx": Lx,
        "Ly": Ly,
        "splits": splits,
    }


def select_composed_family_test_cases(
    geometry_family,
    case_ids,
    original_composed_indices,
    selected_families,
    requested_case_numbers,
):
    """
    Select zero-based positions within each family's exact saved test subset.

    Missing mapping entries mean "all test cases" for that selected family.
    """
    geometry_family = np.asarray(geometry_family, dtype=object)
    case_ids = np.asarray(case_ids, dtype=np.int64)
    original_composed_indices = np.asarray(
        original_composed_indices,
        dtype=np.int64,
    )
    requested = {
        canonical_geometry_name(family): tuple(int(value) for value in values)
        for family, values in dict(requested_case_numbers).items()
    }
    extra = sorted(set(requested) - set(selected_families))
    if extra:
        raise ValueError(
            "composed_test_case_numbers contains families not selected for "
            f"evaluation: {extra}"
        )

    selected_parts = []
    selection_records = {}
    for family in selected_families:
        family_indices = np.flatnonzero(geometry_family == family)
        if len(family_indices) == 0:
            continue
        if family not in requested:
            chosen_positions = tuple(range(len(family_indices)))
        else:
            chosen_positions = requested[family]
            if len(chosen_positions) == 0:
                raise ValueError(
                    f"Explicit composed test selection for {family!r} is empty."
                )
            if len(chosen_positions) != len(set(chosen_positions)):
                raise ValueError(
                    f"Composed test selection for {family!r} contains duplicates."
                )
            invalid = [
                value
                for value in chosen_positions
                if value < 0 or value >= len(family_indices)
            ]
            if invalid:
                raise IndexError(
                    f"Composed {family} test-case positions {invalid} are "
                    f"outside [0, {len(family_indices) - 1}]."
                )
        chosen_indices = family_indices[np.asarray(chosen_positions, dtype=int)]
        selected_parts.append(chosen_indices)
        selection_records[family] = [
            {
                "family_test_case_number": int(position),
                "composed_case_id": int(case_ids[index]),
                "original_composed_index": int(
                    original_composed_indices[index]
                ),
            }
            for position, index in zip(chosen_positions, chosen_indices)
        ]

    if not selected_parts:
        raise ValueError("The saved test split contains none of the selected families.")
    return np.concatenate(selected_parts), selection_records


def evaluate_composed_test_dataset(cfg):
    """Evaluate only unseen simple-family cases from the saved test split."""
    if str(cfg.training_dataset_mode).strip().lower() != "composed":
        raise ValueError(
            "Composed test evaluation requires training_dataset_mode='composed'."
        )
    if not os.path.isfile(cfg.model_path):
        raise FileNotFoundError(f"Trained model does not exist: {cfg.model_path}")
    os.makedirs(cfg.composed_evaluation_root, exist_ok=True)

    selected_families = tuple(
        canonical_geometry_name(value)
        for value in cfg.composed_evaluation_families
    )
    invalid = sorted(set(selected_families) - set(TRAINING_GEOMETRY_FAMILIES))
    if invalid:
        raise ValueError(f"Unsupported composed evaluation families: {invalid}")
    if not selected_families:
        raise ValueError("No composed simple families are selected.")

    device = get_device(cfg)
    scaler = load_training_temperature_scaler(cfg)
    training_settings = load_training_model_settings(cfg)
    dataset = load_composed_test_dataset(cfg, training_settings, scaler)

    requested_case_numbers = dict(cfg.composed_test_case_numbers)
    selected_indices, selection_records = select_composed_family_test_cases(
        dataset["geometry_family"],
        dataset["case_ids"],
        dataset["original_composed_indices"],
        selected_families,
        requested_case_numbers,
    )
    for key in (
        "sdf",
        "mask",
        "temp_physical",
        "geometry_family",
        "fe_paths",
        "metadata_json",
        "case_ids",
        "source_case_ids",
        "candidate_ids",
        "original_composed_indices",
    ):
        dataset[key] = dataset[key][selected_indices]
    dataset["ncases"] = len(selected_indices)

    model = SimpleDeepONet(
        branch_in=dataset["Nx"] * dataset["Ny"],
        trunk_in=3,
        latent=training_settings["latent_dim"],
    ).to(device)
    try:
        state_dict = torch.load(
            cfg.model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    temperature_min = scaler["temperature_min"]
    temperature_max = scaler["temperature_max"]
    temperature_range = temperature_max - temperature_min
    truth_normalized = apply_temperature_normalization(
        dataset["temp_physical"],
        dataset["mask"],
        temperature_min,
        temperature_max,
    )
    prediction_normalized = predict_ood_dataset(
        model,
        dataset,
        device,
        cfg.ood_evaluation_batch_size,
    )
    # Match the legacy evaluator: retain the unmasked prediction for
    # Cartesian-to-FE interpolation. Plotting and metrics apply the material
    # mask separately, so hollow-region zeros cannot contaminate FE vertices.
    prediction_physical = (
        prediction_normalized * temperature_range + temperature_min
    )

    all_rows = []
    family_summaries = {}
    for family in selected_families:
        family_case_numbers = np.flatnonzero(
            dataset["geometry_family"] == family
        )
        if len(family_case_numbers) == 0:
            continue
        family_dir = cfg.composed_evaluation_family_dir(family)
        os.makedirs(family_dir, exist_ok=True)
        test_number_by_case_id = {
            int(record["composed_case_id"]): int(
                record["family_test_case_number"]
            )
            for record in selection_records[family]
        }
        if family in {
            canonical_geometry_name(value) for value in requested_case_numbers
        }:
            plot_numbers = set(family_case_numbers)
        else:
            # A missing explicit selection means batch evaluation: calculate
            # every held-out case, but do not create per-case figure bundles.
            plot_numbers = set()
        rows = []
        for case_number in family_case_numbers:
            prediction_grid = prediction_physical[case_number].reshape(
                dataset["Ny"],
                dataset["Nx"],
            )
            fe_result = evaluate_prediction_on_fe_mesh(
                dataset["fe_paths"][case_number],
                prediction_grid,
                dataset["Lx"],
                dataset["Ly"],
            )
            row = compute_case_metrics(
                family=family,
                case_id=dataset["case_ids"][case_number],
                candidate_id=dataset["candidate_ids"][case_number],
                prediction_normalized=prediction_normalized[case_number],
                truth_normalized=truth_normalized[case_number],
                prediction_physical=prediction_physical[case_number],
                truth_physical=dataset["temp_physical"][case_number],
                mask=dataset["mask"][case_number],
                fe_metrics=fe_result["metrics"],
            )
            rows.append(row)
            if case_number in plot_numbers:
                case_id = int(dataset["case_ids"][case_number])
                test_case_number = test_number_by_case_id[case_id]
                case_dir = get_selected_case_output_dir(
                    family_dir,
                    test_case_number,
                    case_id,
                )
                case_dataset = {
                    **dataset,
                    "family": family,
                }
                save_case_figure(
                    cfg,
                    case_dataset,
                    case_number,
                    dataset["temp_physical"],
                    prediction_physical,
                    output_dir=case_dir,
                    evaluation_label="Composed held-out test",
                    save_pdf=cfg.composed_save_pdf_figures,
                )
                save_original_style_fe_outputs(
                    cfg=cfg,
                    output_dir=case_dir,
                    evaluation_label="Composed held-out test",
                    family=family,
                    case_id=case_id,
                    fe_result=fe_result,
                    metrics=row,
                    save_pdf=cfg.composed_save_pdf_figures,
                    geometry_metadata=json.loads(
                        str(dataset["metadata_json"][case_number])
                    ),
                    test_case_number=test_case_number,
                    source_case_id=int(
                        dataset["source_case_ids"][case_number]
                    ),
                    domain={
                        "Lx": dataset["Lx"],
                        "Ly": dataset["Ly"],
                    },
                )
                save_legacy_style_case_bundle(
                    cfg=cfg,
                    dataset=case_dataset,
                    case_number=case_number,
                    test_case_number=test_case_number,
                    truth_physical=dataset["temp_physical"],
                    prediction_physical=prediction_physical,
                    fe_result=fe_result,
                    metrics=row,
                    output_dir=case_dir,
                    evaluation_label="Composed held-out test",
                    save_pdf=cfg.composed_save_pdf_figures,
                )

        save_metrics_csv(os.path.join(family_dir, "metrics.csv"), rows)
        save_rmse_mape_plot_and_csv(
            cfg,
            rows,
            output_root=family_dir,
            filename_prefix=f"all_{family}_test_cases",
            save_pdf=cfg.composed_save_pdf_figures,
        )
        summary = summarize_metrics(rows)
        with open(
            os.path.join(family_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(summary, stream, indent=2)
        all_rows.extend(rows)
        family_summaries[family] = summary
        print(
            f"Composed test {family}: {len(rows)} cases, "
            f"mean FE relative L2={summary['mean_fe_relative_l2']:.6e}"
        )

    combined_metrics_path = os.path.join(
        cfg.composed_evaluation_root,
        "all_composed_test_metrics.csv",
    )
    save_metrics_csv(combined_metrics_path, all_rows)
    save_family_comparison_figure(
        cfg,
        all_rows,
        tuple(family_summaries),
        output_root=cfg.composed_evaluation_root,
        filename_prefix="composed_test",
        save_pdf=cfg.composed_save_pdf_figures,
    )
    payload = {
        "overall": summarize_metrics(all_rows),
        "families": family_summaries,
    }
    with open(
        os.path.join(cfg.composed_evaluation_root, "composed_test_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(payload, stream, indent=2)

    provenance = {
        "evaluation_type": "composed_held_out_simple_test",
        "model_path": cfg.model_path,
        "composed_dataset_path": dataset["path"],
        "split_indices_path": dataset["splits"]["path"],
        "split_method": dataset["splits"]["split_method"],
        "evaluated_original_test_indices": (
            dataset["original_composed_indices"].tolist()
        ),
        "proof_of_exclusion": {
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
            "test_cases_loaded_only_from_saved_test_idx": True,
        },
        "temperature_scaler": scaler,
        "selected_families": list(selected_families),
        "requested_family_test_case_numbers": {
            canonical_geometry_name(family): list(values)
            for family, values in requested_case_numbers.items()
        },
        "resolved_test_case_selection": selection_records,
        "branch_input": "complete flattened SDF field",
        "trunk_input": ["x_normalized", "y_normalized", "sdf(x,y)"],
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "evaluation_config": asdict(cfg),
    }
    with open(
        os.path.join(
            cfg.composed_evaluation_root,
            "evaluation_provenance.json",
        ),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(provenance, stream, indent=2)
    print(f"Saved composed held-out metrics: {combined_metrics_path}")
    return payload
