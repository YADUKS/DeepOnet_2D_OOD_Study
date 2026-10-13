# Migration from the flat study folder

Reference inspected: `prajwal1798/Feature_Extraction`, commit
`6c0d0e14af01d48dadb2fd4b485c7a61c21a205e`. Its repository organisation and
Halton-training/LHS-holdout design informed the initial migration. The subsequent
workflow update adds its DeepONet architecture/activation/LR studies, seed and
dataset-size experiments, reports, timings and checkpoint recovery. FNO and POD
are excluded. Geometry construction and the thermal PDE remain from the original
DeepOnet_2D_OOD_Study.

The Halton routine is an independent implementation of the standard radical
inverse sequence, extended beyond the reference's four dimensions. The later
workflow update adapts reference model/training/runtime/benchmark code and includes
its GPL-3.0 licence and attribution; see the root `ATTRIBUTION.md` and `LICENSE`.

| Previous file(s) | Current location |
|---|---|
| `main_pipeline.py` | Replaced by root `run.py`; original in `archive/legacy/` |
| `config.py` | `src/config.py` |
| `geometry.py`, `geometry_spec.py`, `simple_geometries.py`, `ood_geometries.py` | `src/fem/`, same basenames |
| `solver.py`, `sampling.py`, `fe_storage.py` | `src/fem/`, same basenames |
| `generate_geometry_datasets.py` | `src/data/generate.py` |
| `generate_ood_datasets.py` | `src/data/external.py` |
| `compose_geometry_dataset.py` | `src/data/compose.py` |
| `training_data.py`, `ml_dataset.py` | `src/data/protocol.py`, `src/data/dataset.py` |
| `model.py` | `src/models/deeponet.py` |
| `train_deeponet.py` | `src/training/trainer.py` |
| `evaluate_composed.py`, `evaluate_ood.py` | Initially retained as `src/analysis/evaluate.py`, `src/analysis/ood.py`; active CLI now uses `src/analysis/metrics.py` |
| `preview_stored_geometry.py` | `src/analysis/preview.py` |
| `export_ansys_geometry.py` | `src/analysis/export_ansys.py` |
| `plot_style.py`, `plot_utils.py` | `src/analysis/style.py`, `src/analysis/plots.py` |
| `generate_dataset.py`, `generate_rounded_rect_test_dataset.py` | `src/data/legacy.py`, `src/data/rounded.py` (historical helpers) |
| `evaluate_deeponet.py`, `evaluate_all_test_cases.py` | `src/analysis/legacy.py`, `src/analysis/all_legacy_cases.py` |
| `evaluate_rounded_rect_test_cases.py`, `predict_compare_case.py` | `src/analysis/rounded.py`, `src/analysis/predict.py` |
| `plot_sampling_space.py` | `src/analysis/legacy_sampling.py`; new multi-family plots in `src/analysis/sampling.py` |

New modules `src/data/design.py` and `src/data/generation.py` centralise
parameter designs, provenance, duplicate handling and role-labelled FE storage.
`src/training/runtime.py` provides one explicit CPU/CUDA choice for the active
training/evaluation paths. Package imports replace working-directory imports.

## Scientific changes

The old study generated one oversampled LHS per family, randomly selected cases
during composition, and randomly split the composition within each family.
The new protocol generates independent LHS holdouts and a nested Halton training
pool. Composition retains roles and chooses training prefixes. This changes the
study data, so numerical results from the old and new protocols are different
experiments. Physical parameters, valid shape ranges and the SDF sign are retained.
The active model now has configurable branch/trunk depth, width and activation.
Temperature scaling uses solid points from the entire maximum training pool;
SDF mean/std use the selected training prefix. Neither uses holdout/OOD values.

Defaults now consistently point to 800 training-pool cases per family, plus
50 validation and 50 test cases. The source previously generated 1000 cases by
default but selected 500-case source files for composition. Family generation
defaults now include all nine simple families. The original configuration is
preserved in the external source backup. The size study uses 50/100/200/400/800
cases per family from one master dataset, with fixed validation/test sets.

Generation rejects nonfinite values at any solid grid point or FE vertex. The
older generator could replace some invalid temperatures with zero. Exact
geometries are rejected across roles using canonical boundary identities,
including circle/ellipse symmetries and polygon vertex ordering. Failed candidates are
recorded in the manifest; an exhausted candidate plan raises an error.

The `train_fraction` and `val_fraction` settings now apply only to retained
legacy helpers. The active protocol uses explicit per-family holdout counts.
Composition and design IDs are independent of the repository's absolute path.
Evaluation verifies the trained split membership; the saved scaler also records
the training dataset's SHA-256 digest. Old model checkpoints cannot resume the
new workflow. Completed FE cases are cached independently of training-pool size.

New training geometries register mapper/boundary callbacks in
`src/fem/custom_shapes.py`; see [adding_geometry.md](adding_geometry.md).

## Verification

Use `python tests/run_all.py` for sampling, geometry, composition, normalisation,
relocation, model gradients and FE heat-balance tests. Use `python run.py smoke`
for a small real FE-to-training-to-evaluation workflow. These tests run the FE
and Torch groups in separate processes because native OpenMP libraries in some
Windows installations conflict when imported together in the opposite order.

CPU/GPU Slurm wrappers are templates based on the reference's queue choices.
Their account, modules, resources and environment must match the actual cluster
allocation. Local verification cannot certify a future Sunbird deployment.
The completed checks are recorded in [verification.md](verification.md).
