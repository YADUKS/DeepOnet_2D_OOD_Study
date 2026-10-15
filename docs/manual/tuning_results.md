# Results from the completed Tanh grid

`report-tuning` reads the completed grid and evaluates its validation-selected
checkpoint. It does not train networks, regenerate geometries or solve FE cases.
Use the same `study.json`, data, tuning size, seeds and training settings that
produced the grid. All configured cells and validation summaries must exist.
The configured OOD sources must already have been generated with `generate-ood`.

## Sunbird: two jobs after the 25 cells have finished

From the study repository in PuTTY or VS Code's remote terminal:

```bash
cd /scratch/s.y.k.sukumarapillai/DeepOnet_2D_OOD_Study
export STUDY_ENV="$PWD/env"
export STUDY_PYTHON="$STUDY_ENV/bin/python"
export STUDY_ACCOUNT="scw7010"

REPORT_EVAL=$(sbatch --parsable --account="$STUDY_ACCOUNT" \
  jobs/sunbird/gpu.sh report-tuning --config study.json \
  --stage-name grid --report-phase evaluate)
echo "Report evaluation job: $REPORT_EVAL"

REPORT_FIGURES=$(sbatch --parsable --account="$STUDY_ACCOUNT" \
  --dependency="afterok:$REPORT_EVAL" \
  jobs/sunbird/cpu.sh report-tuning --config study.json \
  --stage-name grid --report-phase figures)
echo "Report figures job: $REPORT_FIGURES"
```

The GPU job checks all saved grid results, selects the existing validation winner
and caches its predictions for every ID test case and every case in the selected
OOD source families. The CPU job creates all figures. No `tune-select` submission
is needed beforehand: the report runs the same selection and writes the usual
selection JSON for the subsequent study stages.

Do not attach a dependency on an old generation/grid job that has already left
Slurm's active records. The commands above assume those stages have succeeded.
If integrating this into a newly submitted workflow, the evaluation job can use
`--dependency="afterok:$TANH:$OOD"` while those IDs are still valid.

Watch the jobs with:

```bash
squeue -u "$USER"
tail -n 50 -F "outputs/slurm/ood_gpu_${REPORT_EVAL}.out"
# Ctrl+C stops tail; the submitted job continues.
tail -n 50 -F "outputs/slurm/ood_cpu_${REPORT_FIGURES}.out"
sacct -j "$REPORT_EVAL,$REPORT_FIGURES" --format=JobID,State,ExitCode,Elapsed
```

In a new login the shell variables are gone. Substitute the actual job number in
the filename, or assign `REPORT_FIGURES=JOB_NUMBER` before using the command.
Slurm jobs use the explicit interpreter above; environment activation is optional.
If evaluation fails, inspect its log, fix that error and resubmit both jobs with
the new evaluation ID. An `afterok` dependency cannot run after a failed job.

## Windows / WSL / local VS Code

With the project's environment active, use the same command on either platform:

```text
python run.py report-tuning --config study.json --stage-name grid
```

This runs evaluation and plotting sequentially. `--device cpu` permits CPU
inference. `--stage-name grid_gelu` generates the same package for the GELU grid
in a separate directory. Detailed reports are currently for activation grids,
not the three learning-rate cells.

To plot again using the cached report only:

```text
python run.py report-tuning --config study.json --stage-name grid --report-phase figures
```

You can copy the whole report folder from Sunbird and render it locally, without
copying the original datasets or checkpoints:

```text
python run.py report-tuning --report-phase figures --report-dir PATH_TO_COPIED_REPORT --dpi 300
```

Quote `PATH_TO_COPIED_REPORT` if it contains spaces. `--no-pdf` saves only PNGs.
The default is PNG at `publish_dpi` (600) plus a separate PDF for every figure.
Changing plot settings does not change cached predictions or metrics.

## Files produced

The default root is:

```text
outputs/reports/design_space_<composition ID>/tanh_grid/
  N0200_per_family/seeds_123/model_seed123/
```

The output root, tuning size and seeds follow the actual configuration. The job
prints the full directory. An optional `--report-dir PATH` chooses another root;
pass it to both evaluation and figures jobs.

```text
README.md                         # clickable inventory and metric definitions
report_manifest.json              # selection, settings, hashes and completion
figure_index.json                 # every figure's paths and description
01_grid_maps/
  validation_normalized_mse.png    # every name also has its own PDF
  validation_rmse_K.png
  validation_mae_K.png
  validation_mean_relative_l2.png
02_convergence/
  training_validation_loss.png
  learning_rate_vs_epoch.png
03_in_distribution/<family>/
  errors/<metric>_by_case.png
  contours/best_case_<id>/
    ground_truth_temperature_K.png
    prediction_temperature_K.png
    absolute_error_K.png
    relative_error_percent.png
    case_details.json
  contours/median_case_<id>/...    # same four separate figures
  contours/worst_case_<id>/...
04_ood/<family>/...                # same organisation as ID results
05_additional/
  family_distributions/            # separate boxplots by metric and domain
  accuracy_cost/                   # accuracy vs parameters; accuracy vs time
  generalization/                  # ID/OOD RMSE and relative-L2 comparisons
tables/
  grid_runs.csv                    # all cells/seeds, errors, size, epochs, time
  grid_mean_metrics.csv            # the four grid-map values
  selected_loss_history.csv
  selected_epoch_timing.csv        # when present in the original logs
  case_metrics.csv                 # every ID/OOD scenario with geometry identity
  family_summary.csv               # mean/median/p95/max errors per family
  domain_summary.csv               # ID/OOD statistics and pooled-point errors
  representative_cases.csv         # selected best/median/worst IDs and folders
data/<domain>/<family>_predictions.npz
```

There are no combined figure plates in this report. With the default nine ID
families and four OOD families, the report produces **241 individual figures**
(241 PNGs and, unless disabled, 241 PDFs). The report cache stores all evaluated
truth/prediction fields, masks, coordinates, case IDs, geometry keys and metadata;
absolute and relative error fields can be recomputed without inference.

## Reading the results correctly

- **Grid maps:** discrete, labelled layer/neuron cells without smoothing. The
  red outline marks the same validation-selected network on all four maps.
  Validation normalized MSE, pooled-point RMSE/MAE and mean case-relative L2 are
  different metrics; the winner uses the existing mean case-relative L2 rule.
- **Selection:** each network's best epoch minimizes validation MSE. Architecture
  selection minimizes mean validation case-relative L2 in Kelvin. Test and OOD
  values are not used to select an architecture, epoch or model seed.
- **Multiple tuning seeds:** grid cells show arithmetic means across those seeds.
  Predictions use the selected architecture at the first configured tuning seed,
  or the saved seed explicitly specified by `--report-seed`. Pass the same
  `--seeds` list used for tuning/selection to both report jobs. There is no test-based
  seed selection and no ensemble inference.
- **Loss history:** the saved training and validation curves are plotted together.
  The held-out ID test loss is shown as one star at the selected epoch. Existing
  runs did not save per-epoch test losses; a test curve cannot be recovered from
  the best checkpoint alone.
- **Case ranking:** best/median/worst are the lowest, middle and highest relative-L2
  test cases **within each family**, using this one selected model. Even-sized
  families use the lower median; equal errors are ordered by original case ID.
  Small families can have the same case in more than one role. Per-family curves
  preserve original scenario order rather than sorting their x-axis by error.
- **Temperature and errors:** grid fields use solid points only. Cavities are
  masked. Ground truth and prediction share a colour range across all three
  representatives within a family. Each error type also shares a colour range
  across those representatives. Different families have separate colour ranges.
  These are the saved Cartesian evaluation fields, not new FE-mesh solutions.
- **Relative L2:** `||prediction - truth|| / ||truth||`, with temperatures in Kelvin.
  Case tables store fractions; figures display percentages. Temperature-rise L2
  instead divides by `||truth - T_inf||`; a near-zero temperature rise can give a
  large ratio. Pointwise relative-error maps use
  `100 * abs(prediction - truth) / max(abs(truth), 1e-12)`.
- **Peak error:** signed `max(prediction) - max(truth)` is in the case table; its
  absolute value is plotted for each scenario. It measures maximum-temperature
  error, not the spatial distance between hot spots.
- **Timing:** accuracy/cost uses the recorded cumulative training plus validation
  time, and omits cells without complete timing history. Concurrent cluster
  jobs and hardware differences can affect observed timings.
- **Generalization plots:** ID/OOD comparisons pool cases descriptively; different
  geometry families are not paired experiments or confidence intervals.

## Integrity and repeated runs

Evaluation rejects missing grid cells, mismatched data/settings, stale checkpoint
metrics, incomplete validation coverage and overlapping or incompatible OOD
sources. The cache manifest records dataset, checkpoint, grid-table/history and
OOD fingerprints. Repeating evaluation with unchanged inputs reuses the verified
predictions. If inputs change, use a new `--report-dir` rather than mixing studies.
The figures phase checks cached file hashes before rendering and requires no
Torch, NGSolve or CUDA imports. Do not run two writers into the same report root.

This is a diagnostic report for the Tanh study. Keep later activation/LR choices
based on their validation criteria, even after viewing these test/OOD results.
