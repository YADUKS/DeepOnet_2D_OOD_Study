# Earlier workflow

`legacy/` preserves the original switch-based entry point, README and frozen
environment list. It is a historical reference, not the supported launcher.
Use `python run.py --help` from the repository root.

`pre_study_workflow/` preserves the model, data loader, trainer and runtime from
the intermediate restructuring, before the configurable reference-style study.
The supported CLI now uses v3 study checkpoints; these older model files are
historical references and cannot resume current runs.

The complete pre-migration source was also backed up outside this repository
in the parent workspace's `migration_backups/` folder. Old generated LHS datasets
and weights were not converted to the new sampling design. Keep them with their
original configuration and source when reproducing earlier results.

Retained legacy helpers in `src/data/` and `src/analysis/` support inspection of
the old circle/rectangle and rounded-rectangle code. They are not dispatched by
the current OOD study CLI and are not inputs to its v2 composition protocol.
