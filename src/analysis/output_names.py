"""Readable output labels and explicit, non-overwriting legacy folder renames."""
from pathlib import Path

STAGE_NAMES = {'grid': 'tanh_neurons_layers_grid', 'grid_gelu': 'gelu_neurons_layers_grid',
               'lr': 'learning_rate_study'}
CATEGORIES = ('checkpoints', 'logs', 'study_results', 'figures', 'reports')


def stage_name(stage):
    return STAGE_NAMES[stage]


def rename_outputs(cfg, *, dry_run=False):
    """Rename this composition's output roots; never touch dataset/cache contents.

    All destinations are checked before the first move. Existing readable roots
    are not merged or overwritten. Stop running jobs before invoking this.
    """
    root = Path(cfg.output_root).resolve()
    plan = []
    if cfg.case_tag == cfg.legacy_case_tag:
        print('The configured study uses legacy names; no folders to rename.')
        return plan
    for category in CATEGORIES:
        parent = root/category
        source = parent/cfg.legacy_case_tag
        destination = parent/cfg.case_tag
        # Check resolved absolute paths before any directory move, including
        # Windows junctions or symbolic links that lead outside the output root.
        if (parent.resolve().parent != root or source.resolve().parent != parent.resolve()
                or destination.resolve().parent != parent.resolve()):
            raise ValueError(f'Output rename would leave the intended directory: {parent}')
        if source.is_symlink() or destination.is_symlink():
            raise ValueError(f'Refusing to rename linked output roots: {source}')
        if not source.exists():
            continue
        if not source.is_dir():
            raise ValueError(f'Expected an output directory: {source}')
        if destination.exists():
            raise FileExistsError(f'Both output roots exist; nothing was renamed: {source} and {destination}. '
                                 'Choose a different --study-name or keep automatic legacy lookup.')
        plan.append((source, destination))
    for source, destination in plan:
        print(f'{source}\n  -> {destination}', flush=True)
    if dry_run:
        print(f'Preview only: {len(plan)} folders; no files changed.')
        return [(str(a), str(b)) for a, b in plan]
    moved = []
    try:
        for source, destination in plan:
            if destination.exists():
                raise FileExistsError(f'Rename destination appeared: {destination}')
            source.rename(destination)
            moved.append((source, destination))
    except OSError:
        # Earlier renames are reversible because contents/relative paths never
        # change. Do not overwrite a source recreated by a concurrent writer.
        for source, destination in reversed(moved):
            if source.exists():
                raise RuntimeError(f'Concurrent output writer detected; renamed files remain at {destination}.')
            destination.rename(source)
        raise
    print(f'Renamed {len(moved)} output folders. Datasets and file contents were preserved.', flush=True)
    return [(str(a), str(b)) for a, b in plan]
