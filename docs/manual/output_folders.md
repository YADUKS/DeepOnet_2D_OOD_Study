# Readable study folders

Set a study label in `study.json`:

```json
"study_name": "deeponet_2d_ood"
```

New output roots identify the study, training-family count and maximum training
pool. For the standard nine-family study:

```text
outputs/
  checkpoints/deeponet_2d_ood_9families_pool800_<ID>/<RUN>/
  logs/deeponet_2d_ood_9families_pool800_<ID>/<RUN>/
  study_results/deeponet_2d_ood_9families_pool800_<ID>/<RUN>/
  figures/deeponet_2d_ood_9families_pool800_<ID>/
  reports/deeponet_2d_ood_9families_pool800_<ID>/
```

`pool800` means 800 training cases per included family in the master dataset,
before choosing a smaller tuning/scaling prefix. For unequal family contributions,
the label uses `pooltotalN` instead. `<ID>` is the existing 12-character composition
identity. It remains at the end of the readable name to distinguish different
geometry ranges, family selections, physics, grids or holdouts with the same label.

Names use letters, numbers, underscores and hyphens, start with a letter/number,
and have at most 48 characters. They are converted to lowercase. Choose the name
before creating new outputs and keep it consistent throughout that study. The
CLI also accepts `--study-name my_experiment`; use the same value in subsequent
commands, or store it in `study.json`. An empty name retains the original hash-only
study root names. Existing configuration files without `study_name` use the
default `deeponet_2d_ood`; there is no need to replace their scientific settings.

Changing the label does not change sampling, dataset identity or checkpoint
compatibility. Generated source datasets and FE caches retain their stable data
paths, including `design_<ID>` and `mix_<ID>`; their relative references remain valid.

## Reports identify the study stage

New detailed Tanh reports use:

```text
outputs/reports/deeponet_2d_ood_9families_pool800_<ID>/
  tanh_neurons_layers_grid/
    train200_per_family/
      seeds_123/
        model_seed123/
          README.md
          01_grid_maps/
          02_convergence/
          03_in_distribution/
          04_ood/
          05_additional/
          tables/
          data/
```

GELU uses `gelu_neurons_layers_grid`. `train200_per_family` is the actual tuning
prefix; `seeds_123` is the set of grid seeds, and `model_seed123` identifies the
single checkpoint used for detailed test/OOD predictions. All existing scientific
selection and reporting rules are unchanged.

The existing comparison reports now have separate stage folders under
`outputs/figures/<STUDY>/study_reports/`:

| Folder | Purpose |
|---|---|
| `tanh_neurons_layers_grid/train200_per_family/` | Basic Tanh validation-grid comparison |
| `gelu_neurons_layers_grid/train200_per_family/` | Basic GELU validation-grid comparison |
| `learning_rate_study/train200_per_family/` | Learning-rate comparison |
| `seed_variability_study/train800_per_family/` | Seed and family reports at one training size |
| `dataset_size_study/` | Scaling curves across training sizes and seeds |

The size labels follow the configuration. Existing per-run names such as
`tune_deeponet_N0200_grid_L2_N80_seed123` remain valid; they already identify
training size, grid cell and seed. They are kept so the same completed runs can
be found, skipped or resumed.

## Existing Sunbird results

The code automatically looks for existing model runs under
`outputs/<category>/design_space_<ID>/<RUN>/` when the corresponding readable
run directory does not exist. Old selected-settings JSONs and completed report
caches are also found. You can continue reporting without moving anything.

To rename the existing output roots as well, first let all jobs using this
study's outputs finish. Then, before starting new training or report jobs, run:

```bash
cd /scratch/s.y.k.sukumarapillai/DeepOnet_2D_OOD_Study
export STUDY_ENV="$PWD/env"
export STUDY_PYTHON="$STUDY_ENV/bin/python"

# Review the exact old and new directory names.
"$STUDY_PYTHON" run.py rename-outputs --config study.json --dry-run

# Rename the five output-category roots that exist for this composition.
"$STUDY_PYTHON" run.py rename-outputs --config study.json
```

On Windows/WSL with the project environment active:

```text
python run.py rename-outputs --config study.json --dry-run
python run.py rename-outputs --config study.json
```

This renames parent directories; it does not rewrite datasets, checkpoints,
histories, metrics or cached prediction files. Checksums stay the same. Absolute
paths embedded in old provenance files describe their original locations; they
are not rewritten. Old report subfolders such as `tanh_grid/N0200_per_family`
are preserved and remain discoverable after their study root is renamed. Fresh
reports use the descriptive stage/size names shown above.

The command checks every destination before the first move. By default it refuses
to combine existing roots. Do not rename folders while a queued/running job may
read or write them. No generation or retraining is required for this naming change.

### When both old and readable folders already exist

This can happen when the grid was trained before readable names were added, then
new reporting commands created readable output folders. The plain rename command
stops without moving any files. Automatic legacy lookup remains available, or
you can explicitly consolidate missing outputs into the readable folders:

```bash
./env/bin/python run.py rename-outputs --config study.json --merge-existing --dry-run
./env/bin/python run.py rename-outputs --config study.json --merge-existing
```

Run the second command only after reviewing a successful preview. It:

- Moves entries whose destination does not exist, including missing files inside
  an existing run folder. Existing readable results stay in place.
- Compares matching files by size and SHA-256. Identical files are already
  available in the readable folder; their legacy copies are retained.
- Stops before moving anything if a matching file has different contents or a
  file/directory mismatch. The error lists conflicting paths for review. Neither
  version is overwritten or deleted.
- Rejects linked paths, checks resolved paths stay under the output root, and
  rolls back earlier moves if a later move fails. Jobs must remain stopped.

Only empty legacy directories are removed after successful moves. Legacy roots
can remain if they contain verified identical copies; this is expected. Repeating
the command safely skips these copies. File contents, prediction-cache hashes,
datasets and FE caches are preserved. Names of stage/run subfolders are preserved
as in the plain rename. If conflicts are reported, keep both versions and inspect
those files before choosing what to retain; do not use a forced overwrite.
