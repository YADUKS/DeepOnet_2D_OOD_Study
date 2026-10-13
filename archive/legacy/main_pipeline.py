"""
Main pipeline for the simple-design-space to OOD DeepONet study.

New workflow order:
    1. Generate independent simple-family datasets.
    2. Compose selected families/case counts.
    3. Train with stratified splitting and training-only normalization.
    4. Generate independent OOD datasets.
    5. Evaluate OOD families with the saved training scaler.

The original circle/rectangle workflow remains available in a separate,
explicitly guarded legacy section.
"""

from dataclasses import replace

from compose_geometry_dataset import compose_training_geometry_dataset
from config import Settings
from evaluate_deeponet import evaluate_model
from evaluate_composed import evaluate_composed_test_dataset
from evaluate_ood import evaluate_ood_families
from export_ansys_geometry import export_stored_geometry_for_ansys
from generate_dataset import (
    generate_dataset,
    generate_multiple_datasets,
    run_preview,
)
from generate_geometry_datasets import generate_selected_simple_geometry_datasets
from generate_ood_datasets import generate_selected_ood_datasets
from geometry_spec import (
    OOD_GEOMETRY_FAMILIES,
    TRAINING_GEOMETRY_FAMILIES,
    canonical_geometry_name,
)
from preview_stored_geometry import preview_stored_geometry
from train_deeponet import train_model


# ============================================================
# NEW DESIGN-SPACE WORKFLOW SWITCHES
# ============================================================

RUN_PREVIEW_STORED_GEOMETRY = False
RUN_EXPORT_STORED_GEOMETRY_FOR_ANSYS = False
RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS = False
RUN_COMPOSE_TRAINING_GEOMETRY_DATASET = False
RUN_TRAIN_COMPOSED_MODEL = False
RUN_EVALUATE_COMPOSED_TEST_DATASET = False
RUN_GENERATE_OOD_DATASETS = False
RUN_EVALUATE_OOD_DATASETS = True


# ============================================================
# LEGACY CIRCLE/RECTANGLE WORKFLOW SWITCHES
# ============================================================

RUN_LEGACY_PREVIEW = False
RUN_GENERATE_LEGACY_SELECTED_DATASET = False
RUN_GENERATE_LEGACY_MULTIPLE_DATASETS = False
RUN_TRAIN_LEGACY_MODEL = False
RUN_EVALUATE_LEGACY_MODEL = False


def _canonical_tuple(values):
    return tuple(canonical_geometry_name(value) for value in values)


def _validate_family_selection(values, allowed, label):
    canonical = _canonical_tuple(values)
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{label} contains duplicate geometry families.")
    invalid = sorted(set(canonical) - set(allowed))
    if invalid:
        raise ValueError(f"{label} contains unsupported families: {invalid}")
    return canonical


def validate_pipeline_configuration(cfg):
    """Fail early for contradictory or incomplete workflow selections."""
    mode = str(cfg.training_dataset_mode).strip().lower()
    if mode not in ("legacy", "composed"):
        raise ValueError(
            "training_dataset_mode must be either 'legacy' or 'composed'."
        )

    simple_generation = _validate_family_selection(
        cfg.generate_geometry_families,
        TRAINING_GEOMETRY_FAMILIES,
        "generate_geometry_families",
    )
    ood_generation = _validate_family_selection(
        cfg.generate_ood_families,
        OOD_GEOMETRY_FAMILIES,
        "generate_ood_families",
    )
    ood_evaluation = _validate_family_selection(
        cfg.ood_evaluation_families,
        OOD_GEOMETRY_FAMILIES,
        "ood_evaluation_families",
    )
    composed_evaluation = _validate_family_selection(
        cfg.composed_evaluation_families,
        TRAINING_GEOMETRY_FAMILIES,
        "composed_evaluation_families",
    )

    if RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS and not simple_generation:
        raise ValueError("No simple geometry families are selected for generation.")
    for family in simple_generation:
        if family not in cfg.geometry_cases_per_family:
            raise KeyError(f"No generation count configured for {family!r}.")
        if int(cfg.geometry_cases_per_family[family]) < 0:
            raise ValueError(f"Generation count for {family!r} cannot be negative.")

    selected_training_families = []
    for requested_family, requested_count in cfg.training_geometry_counts.items():
        family = canonical_geometry_name(requested_family)
        if family not in TRAINING_GEOMETRY_FAMILIES:
            raise ValueError(f"Unsupported training family: {family!r}.")
        count = int(requested_count)
        if count < 0:
            raise ValueError(f"Training count for {family!r} cannot be negative.")
        if count == 0:
            continue
        if family not in cfg.training_geometry_source_sizes:
            raise KeyError(f"No source size configured for {family!r}.")
        source_size = int(cfg.training_geometry_source_sizes[family])
        if source_size <= 0:
            raise ValueError(f"Source size for {family!r} must be positive.")
        if count > source_size:
            raise ValueError(
                f"Requested {count} {family} cases from a configured "
                f"{source_size}-case source."
            )
        if count < 3:
            raise ValueError(
                f"Selected family {family!r} needs at least three cases for "
                "stratified train/validation/test splitting."
            )
        selected_training_families.append(family)

    composed_actions = (
        RUN_COMPOSE_TRAINING_GEOMETRY_DATASET
        or RUN_TRAIN_COMPOSED_MODEL
        or RUN_EVALUATE_COMPOSED_TEST_DATASET
        or RUN_EVALUATE_OOD_DATASETS
    )
    if composed_actions:
        if mode != "composed":
            raise ValueError(
                "Composition, composed training/testing, and OOD evaluation require "
                "training_dataset_mode='composed'."
            )
        if not selected_training_families:
            raise ValueError("No positive training geometry counts are selected.")
    if RUN_EVALUATE_COMPOSED_TEST_DATASET and not composed_evaluation:
        raise ValueError(
            "No simple families are selected for composed test evaluation."
        )

    if RUN_PREVIEW_STORED_GEOMETRY:
        preview_group = str(cfg.preview_dataset_group).strip().lower()
        preview_family = canonical_geometry_name(cfg.preview_dataset_family)
        if preview_group == "training":
            if preview_family not in TRAINING_GEOMETRY_FAMILIES:
                raise ValueError(
                    f"Preview family {preview_family!r} is not a training family."
                )
        elif preview_group == "ood":
            if preview_family not in OOD_GEOMETRY_FAMILIES:
                raise ValueError(
                    f"Preview family {preview_family!r} is not an OOD family."
                )
        else:
            raise ValueError(
                "preview_dataset_group must be 'training' or 'ood'."
            )
        if int(cfg.preview_dataset_size) <= 0:
            raise ValueError("preview_dataset_size must be positive.")
        if int(cfg.preview_dataset_case_number) < 0:
            raise ValueError("preview_dataset_case_number cannot be negative.")

    if RUN_EXPORT_STORED_GEOMETRY_FOR_ANSYS:
        export_group = str(cfg.ansys_export_dataset_group).strip().lower()
        export_family = canonical_geometry_name(cfg.ansys_export_family)
        if export_group not in ("training", "composed", "ood"):
            raise ValueError(
                "ansys_export_dataset_group must be 'training', "
                "'composed', or 'ood'."
            )
        if (
            export_group == "training"
            and export_family not in TRAINING_GEOMETRY_FAMILIES
        ):
            raise ValueError(
                f"ANSYS export family {export_family!r} is not a "
                "training family."
            )
        if export_group == "ood" and export_family not in OOD_GEOMETRY_FAMILIES:
            raise ValueError(
                f"ANSYS export family {export_family!r} is not an OOD family."
            )
        if export_group != "composed" and int(cfg.ansys_export_dataset_size) <= 0:
            raise ValueError("ansys_export_dataset_size must be positive.")
        if int(cfg.ansys_export_case_number) < 0:
            raise ValueError("ansys_export_case_number cannot be negative.")
        if float(cfg.ansys_extrusion_thickness) <= 0.0:
            raise ValueError("ansys_extrusion_thickness must be positive.")
        if str(cfg.ansys_export_units).strip().lower() not in ("m", "mm"):
            raise ValueError("ansys_export_units must be 'm' or 'mm'.")

    if RUN_GENERATE_OOD_DATASETS and not ood_generation:
        raise ValueError("No OOD families are selected for generation.")
    for family in ood_generation:
        if family not in cfg.ood_cases_per_family:
            raise KeyError(f"No OOD generation count configured for {family!r}.")
        if int(cfg.ood_cases_per_family[family]) < 0:
            raise ValueError(f"OOD generation count for {family!r} is negative.")

    if RUN_EVALUATE_OOD_DATASETS:
        if not ood_evaluation:
            raise ValueError("No OOD families are selected for evaluation.")
        for family in ood_evaluation:
            if family not in cfg.ood_evaluation_source_sizes:
                raise KeyError(
                    f"No OOD evaluation source size configured for {family!r}."
                )
            if int(cfg.ood_evaluation_source_sizes[family]) <= 0:
                raise ValueError(
                    f"OOD evaluation source size for {family!r} must be positive."
                )

    legacy_actions = (
        RUN_LEGACY_PREVIEW
        or RUN_GENERATE_LEGACY_SELECTED_DATASET
        or RUN_GENERATE_LEGACY_MULTIPLE_DATASETS
        or RUN_TRAIN_LEGACY_MODEL
        or RUN_EVALUATE_LEGACY_MODEL
    )
    if legacy_actions and mode != "legacy":
        raise ValueError(
            "Legacy actions require training_dataset_mode='legacy'."
        )
    if legacy_actions and composed_actions:
        raise ValueError(
            "Do not run legacy and composed workflows in the same invocation."
        )


def _stage(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_pipeline(base_cfg=None):
    """Execute enabled stages in dependency order."""
    cfg = Settings() if base_cfg is None else base_cfg
    validate_pipeline_configuration(cfg)

    enabled = any(
        (
            RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS,
            RUN_PREVIEW_STORED_GEOMETRY,
            RUN_EXPORT_STORED_GEOMETRY_FOR_ANSYS,
            RUN_COMPOSE_TRAINING_GEOMETRY_DATASET,
            RUN_TRAIN_COMPOSED_MODEL,
            RUN_EVALUATE_COMPOSED_TEST_DATASET,
            RUN_GENERATE_OOD_DATASETS,
            RUN_EVALUATE_OOD_DATASETS,
            RUN_LEGACY_PREVIEW,
            RUN_GENERATE_LEGACY_SELECTED_DATASET,
            RUN_GENERATE_LEGACY_MULTIPLE_DATASETS,
            RUN_TRAIN_LEGACY_MODEL,
            RUN_EVALUATE_LEGACY_MODEL,
        )
    )
    if not enabled:
        print("No pipeline stages are enabled. Set the required RUN_* switch to True.")
        return {}

    results = {}

    # ------------------------------------------------------------
    # NEW WORKFLOW
    # ------------------------------------------------------------
    if RUN_PREVIEW_STORED_GEOMETRY:
        _stage("PREVIEW ONE ACTUAL STORED DATASET GEOMETRY")
        results["stored_geometry_preview"] = preview_stored_geometry(cfg)

    if RUN_EXPORT_STORED_GEOMETRY_FOR_ANSYS:
        _stage("EXPORT ONE STORED GEOMETRY FOR ANSYS")
        results["ansys_geometry_export"] = export_stored_geometry_for_ansys(cfg)

    if RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS:
        _stage("1. GENERATE INDEPENDENT SIMPLE-GEOMETRY DATASETS")
        print(f"Families: {cfg.generate_geometry_families}")
        results["simple_datasets"] = (
            generate_selected_simple_geometry_datasets(cfg)
        )

    if RUN_COMPOSE_TRAINING_GEOMETRY_DATASET:
        _stage("2. COMPOSE SELECTED TRAINING GEOMETRIES")
        print(f"Selected counts: {cfg.training_geometry_counts}")
        results["composed_dataset"] = compose_training_geometry_dataset(cfg)

    if RUN_TRAIN_COMPOSED_MODEL:
        _stage("3. TRAIN COMPOSED DESIGN-SPACE MODEL")
        results["training"] = train_model(replace(cfg))

    if RUN_EVALUATE_COMPOSED_TEST_DATASET:
        _stage("4. EVALUATE HELD-OUT SIMPLE-GEOMETRY TEST CASES")
        results["composed_test_evaluation"] = (
            evaluate_composed_test_dataset(cfg)
        )

    if RUN_GENERATE_OOD_DATASETS:
        _stage("5. GENERATE INDEPENDENT OOD DATASETS")
        print(f"Families: {cfg.generate_ood_families}")
        results["ood_datasets"] = generate_selected_ood_datasets(cfg)

    if RUN_EVALUATE_OOD_DATASETS:
        _stage("6. EVALUATE OOD GENERALIZATION")
        print(f"Families: {cfg.ood_evaluation_families}")
        results["ood_evaluation"] = evaluate_ood_families(cfg)

    # ------------------------------------------------------------
    # LEGACY WORKFLOW
    # ------------------------------------------------------------
    if RUN_LEGACY_PREVIEW:
        _stage("LEGACY: PREVIEW CIRCLE/RECTANGLE CASE")
        results["legacy_preview"] = run_preview(cfg)

    if RUN_GENERATE_LEGACY_SELECTED_DATASET:
        _stage("LEGACY: GENERATE SELECTED DATASET")
        legacy_cfg = replace(cfg)
        legacy_cfg.n_cases = legacy_cfg.selected_dataset_size
        results["legacy_dataset"] = generate_dataset(legacy_cfg)

    if RUN_GENERATE_LEGACY_MULTIPLE_DATASETS:
        _stage("LEGACY: GENERATE MULTIPLE DATASETS")
        results["legacy_multiple_datasets"] = generate_multiple_datasets(
            base_cfg=cfg,
            families=cfg.generate_dataset_families,
            sizes=cfg.generate_dataset_sizes,
            preview_first=False,
        )

    if RUN_TRAIN_LEGACY_MODEL:
        _stage("LEGACY: TRAIN MODEL")
        legacy_cfg = replace(cfg)
        legacy_cfg.n_cases = legacy_cfg.selected_dataset_size
        results["legacy_training"] = train_model(legacy_cfg)

    if RUN_EVALUATE_LEGACY_MODEL:
        _stage("LEGACY: EVALUATE MODEL")
        legacy_cfg = replace(cfg)
        legacy_cfg.n_cases = legacy_cfg.selected_dataset_size
        results["legacy_evaluation"] = evaluate_model(legacy_cfg)

    return results


if __name__ == "__main__":
    run_pipeline()
