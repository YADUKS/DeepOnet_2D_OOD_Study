# Study protocol

## One configuration for the complete study

Copy `study.example.json` to `study.json` in the repository root. Use
`--config study.json` for every stage. Paths inside the JSON resolve from the
repository root. Select an identical family list throughout one study.

For each included family, set `geometry_cases_per_family`,
`training_geometry_source_sizes` and `training_geometry_counts` to the maximum
training pool size. Zero training contribution excludes a family from composition.
Use `--families circle,ellipse,...` consistently if selecting via CLI.

`--train-size N` selects N training cases **from every included family** without
changing the saved master dataset. Thus N=50 and nine families means 450 training
cases, plus the unchanged 450 validation and 450 test cases with the defaults.
`--cases` changes the generated maximum pool and all matching source counts;
`--train-count` is the older composition control and should not be used for the
dataset-size study because it creates a different master dataset/scaler.

## Sampling and separation

Training uses unscrambled Halton starting at index 1. Validation and test use
different seeded LHS designs of exactly their requested counts, plus separately
seeded reserve candidates for rejected geometries. Rejection means the accepted
set need not retain exact LHS stratification; inspect `plot-sampling` and manifest
rejection records. Holdouts are accepted first; training duplicates are skipped.

Canonical identities remove bookkeeping, circle/ellipse angular symmetries and
polygon vertex ordering. Coordinates are quantized to 1e-11 m for identity checks;
this is a duplicate screen, not a guarantee that every pair closer than a chosen
distance is rejected. Distinct nearby geometries are allowed. These checks run
before an FE solve, on reuse/composition, and during training/evaluation loading.
Cross-family duplicates are checked too. A copied sample with a new row number
or filename cannot evade the loader check. Saved metadata is used to reconstruct
and verify identities. OOD sets are checked against the whole composed dataset.

Both holdouts use the training families and ranges, with no shared geometry cases.
OOD families remain outside training, validation, tuning and scaler fitting.
Dataset changes invalidate checkpoint and selection fingerprints.

## Tuning and final runs

At `tuning_size` cases per family, run `grid`, `grid_gelu`, `lr`, each followed by
`tune-select`. Both architecture grids use LR 0.001 by default. Hidden depth/width
apply equally to branch and trunk; the final linear projection to latent width 128
is additional to the hidden layers. The LR stage requires both selected grids.

Training uses Adam, masked point-weighted MSE over the solid, FP32, gradient
clipping at norm 1, ReduceLROnPlateau (factor 0.5, patience 200) and early stopping
(patience 800). The best epoch minimizes validation MSE; configuration selection
minimizes the validation mean-case relative L2 using temperatures in kelvin.
RMSE and R² are also recorded. Test and OOD errors never select a winner.

Tuning defaults to seed 123 and 2,000 epochs; final runs use five seeds and 5,000
epochs. Actual runs may stop sooner. `train-scaling` reuses the LR-stage winner at
every size, rather than retuning. `--seeds` overrides the appropriate seed list;
pass the same list to selection/reporting. Missing cells/runs cause a clear error,
rather than a partial winner or silently incomplete scaling curve.

## Restarting and expanding

- `generate`: repeat the same command. Successful FE cases are cached by geometry
  in a design folder independent of maximum training count.
- `tune` and `train-scaling`: repeat the command. Finished matching runs are skipped;
  interrupted runs resume. Changed settings are rejected.
- `train` and `train-final`: add `--resume` with the same run settings. Recovery
  checkpoints save every 50 epochs by default; optimiser, scheduler and Torch/
  loader RNG states are restored. Extending `--epochs` is permitted on manual resume.
- Increasing beyond the maximum pool solves only additional uncached geometries.
  Existing holdouts and training prefixes stay fixed when their settings are unchanged.
  The larger master pool has a new identity and may change the temperature scaler:
  use a new complete set of training runs for comparisons under that larger study.
- Changing geometry/physics/grid/holdout settings creates a new source design.
  Adding a family reuses unchanged family sources but creates a new composition.

Do not submit two writers for the same generation or model run concurrently.
Slurm arrays assign distinct cells or size/seed combinations.

## Reports and files

`TAG=design_space_<composition ID>`; `RUN` includes architecture stage or size/seed.

| Artifact | Location |
|---|---|
| Per-family sources and FE cache | `data/geometry_datasets/FAMILY/design_ID/` |
| Master dataset and manifest | `data/composed_datasets/mix_ID/` |
| Separate OOD sources | `data/ood_datasets/FAMILY/design_ID/` |
| Best and recovery checkpoints | `outputs/checkpoints/TAG/RUN/` |
| Loss history, epoch timings, summary, hardware | `outputs/logs/TAG/RUN/` |
| Validation/test/OOD per-case CSV and summaries | `outputs/study_results/TAG/RUN/` |
| Tuning CSV and selected JSON | `outputs/study_results/TAG/tuning/Nxxxx_per_family/` |
| Individual loss/prediction figures | `outputs/figures/TAG/RUN/` |
| Tuning heatmaps and seed/family reports | `outputs/figures/TAG/report/Nxxxx_per_family/` |
| Scaling PDF/PNG, CSV and fitted rates | `outputs/figures/TAG/report/` |

`summarize` (alias `compare`, now DeepONet-only) reports scaling runs at one size.
`scaling-report` uses log-log least squares to fit error=C*N^(-alpha); N is per
family, and total training counts are recorded in the CSV. Standard deviations
describe training-seed variability on the same test cases, not uncertainty from
resampling test geometries. `plot` does not overwrite full-split metrics.
Family/case-filtered `evaluate` results use `selected_*` filenames so that full
validation/test metrics remain intact. `--case` numbers refer to positions within
each selected family's holdout set; `plot --case` selects one position in the
selected split. Tuning grids and scaling reports reject inconsistent training
budgets/settings or stale checkpoint metrics.

`train-final` additionally supports standalone selected-model runs, named
`final_deeponet_Nxxxx_seedS`; evaluate these by explicit `--run-name`.
