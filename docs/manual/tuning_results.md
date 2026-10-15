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

## Missing OOD data or a generation crash on Sunbird

If evaluation reports `Missing OOD dataset`, first finish `generate-ood` with the
same `study.json`. A file may be absent because generation never completed or
because the existing source belongs to different configuration settings. Keep
the configured physics, geometry ranges and sampling settings unchanged.

The user reported that OOD generation completed after retrying with one CPU and
one thread for OpenBLAS/MKL, following a signal-11 failure. This is the successful
recovery configuration; it does not establish the exact native crash cause.
From the repository, with `STUDY_ENV`, `STUDY_PYTHON` and `STUDY_ACCOUNT` set as
above, submit generation separately:

```bash
OOD=$(sbatch --parsable --account="$STUDY_ACCOUNT" \
  --cpus-per-task=1 \
  --export=ALL,PYTHONFAULTHANDLER=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1 \
  jobs/sunbird/cpu.sh generate-ood --config study.json)
echo "OOD generation job: $OOD"
tail -n 80 -F "outputs/slurm/ood_cpu_${OOD}.out"
# Ctrl+C stops tail. Check the job itself afterwards:
sacct -j "$OOD" --format=JobID,State,ExitCode,Elapsed
```

The CPU launcher passes the one-CPU allocation to NGSolve through `--cpu-threads`.
The environment limits above apply to this generation job; its existing FE cache
is reused. Leave `--force` off when recovering an interrupted run.

A line such as `[star/ood 20/20]` is printed before the final family dataset is
written. Confirm the complete job is `COMPLETED` with exit code `0:0`. If another
crash occurs, retain the log with its enabled Python fault trace for diagnosis.

After successful generation, submit the two report jobs from the first section
again, using a fresh evaluation ID. Cancel obsolete report jobs still showing
`DependencyNeverSatisfied`, using their actual IDs from `squeue -u "$USER"`.
Dependencies on a failed job do not switch to the successful retry automatically.
Completed OOD datasets and trained grid checkpoints can be used immediately;
this documentation update does not require rerunning generation or training.

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

## Choose Cartesian fields, FE vertices, or both

Add this setting to your existing `study.json` to try both methods:

```json
"report_field_methods": ["cartesian", "fe_vertices"]
```

| Setting | Spatial figures generated |
|---|---|
| `["cartesian", "fe_vertices"]` | Both methods, separate figures, plus metric-comparison plots |
| `["cartesian"]` | Original Cartesian-grid fields |
| `["fe_vertices"]` | Direct FE-vertex fields |
| `[]` | Neither set of spatial field figures; common study plots/tables remain |

The example configuration enables both. An existing configuration without this
setting defaults to Cartesian only. This setting applies to `report-tuning`
(Tanh/GELU detailed reports); it does not change training or the legacy
`evaluate`/`plot` commands.

The original Cartesian test evaluation always supplies the study metrics and
best/median/worst case rankings. Disabling its field plots does not replace those
metrics with FE metrics. This keeps completed architecture/seed/size comparisons
consistent and gives both methods the same representative cases.

When FE vertices are enabled, evaluation reads the original FE case files for
every ID/OOD test case and queries the selected saved model at their vertices.
The branch still receives the original Cartesian SDF. The trunk receives the
vertex coordinates and local SDF from the saved cavity boundary, using the
checkpoint's coordinate and SDF scaling. No training, meshing or FE solving is
performed. Missing or inconsistent FE caches cause a clear error rather than
silently falling back to Cartesian interpolation.

The new query helper is checked against the original helper at identical
Cartesian points before FE inference. FE results have their own manifest,
source-mesh checksums, cache and tables. Existing Cartesian reports can be
extended in place: their original predictions, metric tables and rankings are
verified and preserved, then FE results are added.

After enabling FE vertices for the first time, run the evaluation job followed
by the figures job from the first section. Once the FE cache exists, change the
list and rerun only the CPU figures job:

```bash
sbatch --account="scw7010" jobs/sunbird/cpu.sh report-tuning \
  --config study.json --stage-name grid --report-phase figures
```

This allows switching between either method or both without further inference.
Previously written figures are retained when a method is disabled; the updated
`README.md`, `figure_index.json` and `rendered_field_methods` in the main manifest
identify the figures produced by the current render.

**Colour scales:** each case uses limits derived only from its enabled
ground-truth fields. Ground truth and prediction share exactly the same limits,
colour map and ticks. With both methods, the range spans the union of Cartesian
and FE reference temperatures for that case, so all four temperature figures
use the same scale. Predicted values never expand it. Blue/green under/over
colours, colour-bar extensions and printed field minima/maxima expose excursions
outside the reference range. Absolute and relative error maps have separate
scales, shared across methods for the same case.

**Accuracy comparison:** `tables/case_method_comparison.csv` puts both methods'
RMSE, MAE, relative L2, temperature-rise L2 and peak errors alongside each other
for every case, including their point counts. The FE metrics use equal weights
at FE vertices; these are not area-weighted FE norms. Their differences from
Cartesian metrics reflect different evaluation points and sampling density.
They do not establish a change in the trained model or pointwise differences
at identical coordinates. The paired-case RMSE/L2 scatter plots are descriptive.

FE contours use the original triangle connectivity, leaving the actual cavity
empty. They linearly interpolate the saved/reference and predicted vertex
temperatures inside each triangle. The saved vertex cache does not reconstruct
the full quadratic FE field between vertices. Rankings continue to use the
original Cartesian relative L2 throughout.

## Files produced

The default root is:

```text
outputs/reports/deeponet_2d_ood_9families_pool800_<composition ID>/
  tanh_neurons_layers_grid/train200_per_family/seeds_123/model_seed123/
```

The output root, tuning size and seeds follow the actual configuration. The job
prints the full directory. An optional `--report-dir PATH` chooses another root;
pass it to both evaluation and figures jobs.

`study_name` controls the readable study prefix. Existing hash-only roots and
older `tanh_grid/N0200_per_family` report caches are found automatically, so
their original paths can still appear when reusing completed results. To rename
existing output roots without rewriting scientific files, see the
[folder naming guide](output_folders.md).

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
  contours/cartesian/best_case_<id>/
    ground_truth_temperature_K.png
    prediction_temperature_K.png
    absolute_error_K.png
    relative_error_percent.png
    case_details.json
  contours/cartesian/median_case_<id>/...    # same four separate figures
  contours/cartesian/worst_case_<id>/...
  contours/fe_vertices/<rank>_case_<id>/...  # the same four files, when enabled
04_ood/<family>/...                # same organisation as ID results
05_additional/
  family_distributions/            # separate boxplots by metric and domain
  accuracy_cost/                   # accuracy vs parameters; accuracy vs time
  generalization/                  # ID/OOD RMSE and relative-L2 comparisons
06_method_comparison/              # RMSE/L2 scatter plots when both are enabled
tables/
  grid_runs.csv                    # all cells/seeds, errors, size, epochs, time
  grid_mean_metrics.csv            # the four grid-map values
  selected_loss_history.csv
  selected_epoch_timing.csv        # when present in the original logs
  case_metrics.csv                 # every ID/OOD scenario with geometry identity
  family_summary.csv               # mean/median/p95/max errors per family
  domain_summary.csv               # ID/OOD statistics and pooled-point errors
  representative_cases.csv         # selected best/median/worst IDs and folders
  fe_vertices_case_metrics.csv      # optional: equal-weight FE vertex errors
  fe_vertices_family_summary.csv    # optional: FE family statistics
  case_method_comparison.csv        # optional: Cartesian/FE metrics for every case
data/<domain>/<family>_predictions.npz
data/fe_vertices/<domain>/<family>/case_<id>.npz
fe_vertices_manifest.json          # optional: mesh hashes and query-helper check
```

There are no combined figure plates in this report. With the default nine ID
families and four OOD families, either single field method produces **241 individual
figures**; enabling both produces **399**. Each has a PNG and, unless disabled,
a separate PDF. The report cache stores all evaluated
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
- **Temperature and errors:** Cartesian fields mask cavity cells; FE figures use
  the original mesh triangles. Temperature limits use ground truth only for each
  case, shared across enabled methods and their predictions. Error ranges are
  also shared across methods for that case. Different cases can have different
  ranges. FE evaluation adds queries to the frozen model and never solves a new
  FE problem. See the method-choice section above.
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
