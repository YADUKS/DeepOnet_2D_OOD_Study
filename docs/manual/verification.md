# Study workflow verification — 13 October 2026

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
| Visual checks | Inspected the three-metric tuning heatmap, scaling plot and six-panel custom-shape preview |

The CPU smoke uses only one grid cell per activation, one LR, two seeds and two
epochs. The default 25-cell activation grids are checked structurally in the
regression suite; the full 53 tuning and 25 final training runs were not launched.
These small runs verify execution, not final research accuracy or a sufficient
production mesh/grid resolution. WSL execution and VS Code debugging were not
exercised. The user verified Sunbird login-node imports after repairing the
native runtime; a successful complete smoke on a Sunbird compute node is still
pending. Cluster allocations and environment settings must be supplied at
deployment.

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
