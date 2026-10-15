"""Readable output labels and explicit, non-overwriting legacy folder renames."""
from pathlib import Path

STAGE_NAMES = {'grid': 'tanh_neurons_layers_grid', 'grid_gelu': 'gelu_neurons_layers_grid',
               'lr': 'learning_rate_study'}
CATEGORIES = ('checkpoints', 'logs', 'study_results', 'figures', 'reports')


def stage_name(stage):
    return STAGE_NAMES[stage]


def _check_child(path, parent, root):
    """Reject links and resolved paths outside the configured output root."""
    if (not parent.resolve().is_relative_to(root)
            or path.resolve().parent != parent.resolve()):
        raise ValueError(f'Output rename would leave the intended directory: {path}')
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError(f'Refusing to rename linked output paths: {path}')


def _check_tree(path, root, directories):
    _check_child(path, path.parent, root)
    if path.is_dir():
        directories.append(path)
        for child in path.iterdir():
            _check_tree(child, root, directories)
    elif not path.is_file():
        raise ValueError(f'Expected an ordinary output file or directory: {path}')


def _merge_plan(source, destination, root, moves, duplicates, conflicts):
    from src.data.design import file_digest

    _check_child(destination, destination.parent, root)
    if not destination.exists():
        moves.append((source, destination))
    elif source.is_dir() and destination.is_dir():
        for child in sorted(source.iterdir()):
            _merge_plan(child, destination/child.name, root, moves, duplicates, conflicts)
    elif source.is_file() and destination.is_file():
        if source.stat().st_size == destination.stat().st_size and file_digest(source) == file_digest(destination):
            duplicates.append((source, destination))
        else:
            conflicts.append(f'Different contents: {source}\n  and {destination}')
    else:
        conflicts.append(f'File/directory mismatch: {source}\n  and {destination}')


def rename_outputs(cfg, *, dry_run=False, merge_existing=False):
    """Rename this composition's output roots; never touch dataset/cache contents.

    All destinations are checked before the first move. Explicit merge mode
    moves missing entries and retains identical legacy copies. Different files
    at the same relative path stop the entire operation. No files are deleted
    or overwritten. Stop running jobs before invoking this.
    """
    root = Path(cfg.output_root).resolve()
    plan = []
    duplicates, conflicts, directories = [], [], []
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
            if not merge_existing:
                raise FileExistsError(f'Both output roots exist; nothing was renamed: {source} and {destination}. '
                                     'Review rename-outputs --merge-existing --dry-run, or keep automatic legacy lookup.')
        if merge_existing:
            _check_tree(source, root, directories)
            _merge_plan(source, destination, root, plan, duplicates, conflicts)
        else:
            plan.append((source, destination))
    if conflicts:
        detail = '\n'.join(conflicts[:20])
        if len(conflicts) > 20:
            detail += f'\n... and {len(conflicts)-20} more conflicts.'
        raise FileExistsError(f'{len(conflicts)} conflicting output paths; nothing was renamed. '
                             f'Keep both versions for review; no overwrite is allowed.\n{detail}')
    for source, destination in plan:
        print(f'{source}\n  -> {destination}', flush=True)
    if duplicates:
        print(f'{len(duplicates)} identical files verified by SHA-256; legacy copies will be retained.', flush=True)
    if dry_run:
        print(f'Preview only: {len(plan)} {"entries" if merge_existing else "folders"}; no files changed.')
        return [(str(a), str(b)) for a, b in plan]
    moved = []
    try:
        for source, destination in plan:
            if merge_existing:
                _check_child(source, source.parent, root)
                _check_child(destination, destination.parent, root)
            if destination.exists():
                raise FileExistsError(f'Rename destination appeared: {destination}')
            source.rename(destination)
            moved.append((source, destination))
    except (OSError, ValueError):
        # Earlier renames are reversible because contents/relative paths never
        # change. Do not overwrite a source recreated by a concurrent writer.
        for source, destination in reversed(moved):
            if merge_existing:
                _check_child(source, source.parent, root)
                _check_child(destination, destination.parent, root)
            if source.exists():
                raise RuntimeError(f'Concurrent output writer detected; renamed files remain at {destination}.')
            destination.rename(source)
        raise
    # Only remove empty legacy directories after every move has succeeded.
    # Identical legacy files remain available as copies, never unlinked.
    for path in sorted(set(directories), key=lambda p: len(p.parts), reverse=True):
        if path.exists():
            _check_child(path, path.parent, root)
            try:
                if not any(path.iterdir()):
                    path.rmdir()
            except OSError as error:
                print(f'Output files were moved; could not remove legacy directory {path}: {error}', flush=True)
    print(f'Renamed {len(moved)} output {"entries" if merge_existing else "folders"}. '
          'Datasets and file contents were preserved.', flush=True)
    return [(str(a), str(b)) for a, b in plan]
