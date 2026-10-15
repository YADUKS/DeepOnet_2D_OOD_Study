# Study workflow verification

## Sunbird OOD generation recovery — user reported

OOD generation job 8208460 failed with signal 11 (`ExitCode 0:11`) after logging
20 accepted star cases. That progress count precedes final family dataset writing
and is not a completion check. The user subsequently reported successful
completion after retrying with `--cpus-per-task=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `PYTHONFAULTHANDLER=1`. The retry job ID and its accounting
output were not supplied; the precise native crash cause remains unconfirmed.

The Sunbird manual now uses that configuration for FE generation submissions,
while retaining the CPU launcher's existing allocation for plotting/reporting.
The results guide includes OOD recovery, job-completion checks and replacement
of report dependencies after a failed generation job. This is a documentation
change; solver, sampling, dataset identities and model code are unchanged.

## Completed-grid reporting — 15 October 2026

The full regression suite passed **36 tests** after adding the activation-grid
report: the previous 27, one additional CLI check and eight reporting tests.
The new checks cover a four-cell, two-epoch synthetic-field study with three ID
test cases and three star OOD cases, validation-only selection, masked physical
metrics, weighted validation MAE, within-family ranking, unchanged checkpoints,
missing-cell and stale-table rejection, cache reuse and tamper detection.

The miniature report produced 54 separate figures. A copied report rendered in a
fresh process with imports of Torch, NGSolve and Netgen explicitly blocked, and
without original datasets/checkpoints. This checks portable CPU-only rerendering.
The miniature source fields are analytic test fixtures, not FE research results.

Visual QA inspected the four-cell grid, recorded loss curve, family-error curve,
accuracy/cost scatter and separate ground-truth/prediction/error fields. A
separate synthetic **25-cell** grid confirmed readable discrete cells and the
winner outline at the production grid dimensions. Individual PNG and PDF exports
were generated at 200 dpi for the layout check. The production default remains
600 dpi. The Sunbird CPU/GPU launcher checks and updated Bash guide syntax passed.

Local records (Git ignored):

- `verification/workflow_tests_report.log`: complete 36-test regression run.
- `verification/tuning_report_tests.log`: focused report checks.
- `verification/tuning_report_demo/`: synthetic miniature report and PDF examples.
- `verification/tuning_report_visual_check/`: synthetic 25-cell layout examples.

The user reports that all 25 production Tanh cells have completed on Sunbird.
Those production artifacts are not available in this local checkout, so the new
report has not yet processed the actual Sunbird grid. Run the two jobs in
[the results guide](tuning_results.md) after updating the cluster source. No
production network was retrained and no production data was regenerated here.

## Configurable study workflow — 13 October 2026

The current configurable DeepONet workflow was checked on Windows with the
workspace environment: Python 3.14.3, NumPy 2.4.3, NGSolve 6.2.2602,
PyTorch 2.11.0+cu128 and NVIDIA GeForce RTX 5070 Ti Laptop GPU.

| Current workflow check | Result |
|---|---|
| 27 automated regression tests | Passed: 6 design, 4 protocol, 3 generation/cache/diagnostics, 3 CLI, 1 model, 9 workflow, 1 FE solver |
| Complete CPU study smoke | Passed: generation, composition, Tanh/GELU/LR stages and selection, two sizes × two seeds, summaries, scaling, figures and OOD |
| Geometry separation | Passed: canonical symmetry identities, duplicated geometry with different row IDs rejected at load, duplicate candidate rejected before FE solve |
| FE reuse and extension | Passed: extending a pool solves only new cases; repeating completed generation performs no new solves |
| Generation failure diagnostics | Passed: missing dependencies stop at the first candidate and preserve the diagnostic file; exhausted numerical candidates show the first rejection and full file path |
| Training prefix/scalers | Passed: nested per-family prefixes, fixed holdouts, training-only scalers, full-pool temperature scale frozen across sizes |
| Checkpoint recovery | Passed: resumed CPU training exactly matches uninterrupted weights and loss history; changed run settings rejected |
| Gradient chunking | Passed: chunked and whole-grid updates agree within numerical tolerance |
| All training and OOD shapes | Passed: 36 in-distribution cases across nine families and eight OOD cases across four families |
| GPU workflow | Passed: two-epoch CUDA training, validation/test/OOD evaluation, prediction figures and synchronized benchmark |
| Mesh checks | Passed: three meshes for each of the nine training families, with temperature differences and heat balance recorded |
| Add a training family | Passed: the documented kite example in an isolated checkout generated, composed, mesh-checked, trained, evaluated and previewed |
| Subset evaluation | Passed: selecting one family/case writes separate metrics and leaves the full test summary unchanged |
| Dataset/model relocation | Passed: copied datasets and checkpoint evaluated successfully from another root, including OOD |
| Generation identity stability | Passed: loading data preserves source/OOD identifiers, without rounding physical configuration values |
| Stored FE preview and ANSYS export | Passed with the new FE cache; exported star STL is watertight |
| Platform command files | Sunbird launchers and WSL/Sunbird guides passed Bash syntax checks |
| Sunbird launcher environment | Passed: CPU/GPU interpreter precedence, environment-specific native runtimes, inherited preload entries, venv fallback, missing interpreter and command forwarding (`bash tests/check_sunbird_launchers.sh`, using host-library stand-ins under Git Bash) |
| Sunbird login-node imports (user reported) | NGSolve 6.2.2406 and PyTorch 2.6.0+cu124 import successfully in this study's own Python 3.11 environment with its Conda libgcc/libstdc++ preloaded; package installation succeeded with the cache on scratch |
| Sunbird compute-node CPU smoke (user reported) | Passed: job 8207500 was COMPLETED with exit code 0:0 in 73 seconds; generation, miniature tuning/scaling, validation/test and OOD evaluation finished with `Smoke test passed` |
| Sunbird GPU compatibility (user reported) | Passed: gpu-info job 8207507 was COMPLETED with exit code 0:0 in 3 seconds on scs2041; CUDA initialization and a small matrix multiplication succeeded on NVIDIA A100-PCIE-40GB with driver 550.54.14 and PyTorch 2.6.0+cu124 |
| Visual checks | Inspected the three-metric tuning heatmap, scaling plot and six-panel custom-shape preview |

The CPU smoke uses only one grid cell per activation, one LR, two seeds and two
epochs. The default 25-cell activation grids are checked structurally in the
regression suite; the full 53 tuning and 25 final training runs were not launched.
These small runs verify execution, not final research accuracy or a sufficient
production mesh/grid resolution. WSL execution and VS Code debugging were not
exercised. The user verified Sunbird login-node imports and the complete CPU
smoke after repairing the native runtime. GPU compatibility was subsequently
verified on an allocated A100 node. Full Sunbird GPU training and the production
study remain unverified. The CUDA warning on the CPU smoke host does not describe
the checked GPU node, whose driver is 550.54.14. Cluster allocations and
environment settings must be supplied at deployment.

The verified Sunbird GPU environment uses Python 3.11.17, NumPy 2.2.6,
Matplotlib 3.11.2, NGSolve/Netgen 6.2.2406 and PyTorch 2.6.0+cu124 on
Linux x86_64 with glibc 2.17. CUDA reports available, TF32 is disabled and
deterministic Torch algorithms are enabled. The GPU check exercises device
initialization and a small matrix multiplication, not training convergence.

Sunbird smoke artifacts are on the cluster at
`/scratch/s.y.k.sukumarapillai/DeepOnet_2D_OOD_Study/verification/study_smoke_1791929947826865155`.

Current local artifacts (Git ignored):

- `verification/workflow_tests_final.log`: regression suite.
- `verification/study_smoke_1791912217552977600/`: complete CPU smoke data/results.
- `verification/workflow_extensions.log` and `verification/workflow_extensions/`:
  nine-family GPU study, OOD, mesh checks, export and isolated kite checkout.
- `verification/workflow_reports_final.log` and `verification/workflow_relocated/`:
  final report/selection checks, subset isolation and relocated evaluation.

The additional pre-workflow source backup is outside this repository:
`../migration_backups/DeepOnet_2D_OOD_Study_before_workflow_20261013_181134.zip`.
No production study was run as part of the local verification.

## Earlier restructuring checks (historical)

The following records the intermediate restructuring before the configurable
workflow above. Its 14-test count and older figure/checkpoint formats describe
that earlier version only.

Verified locally on Windows with the existing workspace Python environment:
Python 3.14.3, NumPy 2.4.3, NGSolve 6.2.2602, PyTorch 2.11.0+cu128,
and an NVIDIA GeForce RTX 5070 Ti Laptop GPU.

| Check | Result |
|---|---|
| 14 automated regression tests | Passed: 6 design/geometry, 4 protocol, 2 CLI selection, 1 model/gradient, 1 FE heat balance |
| Source compilation and local import map | Passed; all 38 source modules resolve their local imports |
| CPU end-to-end smoke | Passed: 16 simple cases, two training epochs, held-out evaluation, two star OOD cases |
| All-family miniature study | Passed: 36 simple cases across nine families; eight OOD cases across four families; composition, training and both evaluations |
| CUDA training | Passed: two-epoch smoke training with saved weights, splits, scaler and histories |
| Repository relocation | Passed: source, datasets and model copied to another root; both evaluators run from an unrelated working directory |
| Selected-family/case evaluation | Passed without changing composition identity; case metrics and full figures saved |
| Stored preview | Passed, including six separate panels |
| ANSYS export | Passed; watertight STL and boundary/physics metadata generated from a stored star |
| Sampling-space plotting | Passed for circle and rectangle sources |
| Sunbird shell scripts | Both CPU and GPU scripts passed Bash syntax checking |
| Visual inspection | Stored six-panel preview and regenerated nine-panel evaluation figure inspected |

These are execution/regression checks on small datasets. Two-epoch networks
are not trained research results, and their errors must not be reported as
the study's final performance. WSL, VS Code's debugger, and live Sunbird jobs
were not exercised. The Sunbird allocation/module/environment settings still
need to match the deployment site.

The combined evaluation figure inherited a layout problem from the original
code. It now uses explicit colour-bar gutters, panel titles and a shared
temperature contour range, verified on a regenerated case. This changes
presentation, not prediction or metric values.

Local verification artifacts and logs are retained under `verification/`,
which is ignored by Git. Main logs are in `verification/logs/`:
`tests_run.log`, `cli_solver_final.log`, `verification_run.log`,
`all_families_pipeline.log`, `gpu_run.log`, `relocation_run.log`,
`preview_run.log`, `export_run.log`, and `selected_case_run.log`.

The pre-migration source ZIP contains all 32 original top-level source and
documentation files and passed a ZIP integrity check. It is outside the new
repository, in the parent workspace's `migration_backups/` directory, named
`DeepOnet_2D_OOD_Study_before_restructure_20261013_163641.zip`.

The directory is prepared for source control with `.gitignore` and
`.gitattributes`. No GitHub repository, commit or upload was created.
