"""Reference DeepONet grid → GELU grid → learning-rate selection, validation only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from src.data.design import file_digest
from src.analysis.metrics import write_rows

STAGES = ('grid', 'grid_gelu', 'lr')
FAIR_SETTINGS = ('epochs', 'batch_size', 'precision', 'point_chunk', 'grad_clip',
                 'early_stop_patience', 'scheduler_patience', 'scheduler_factor', 'train_size')


def tuning_dir(cfg):
    return cfg.study_dir / 'tuning' / f'N{cfg.tuning_size:04d}_per_family'


def winner_path(cfg, stage):
    return tuning_dir(cfg) / f'deeponet_{stage}_selected.json'


def selection(cfg, stage):
    path = winner_path(cfg, stage)
    if not path.exists():
        raise FileNotFoundError(f'Select tuning stage {stage} first: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['dataset_sha256'] != file_digest(cfg.dataset_path):
        raise ValueError('Tuning winner belongs to a different dataset.')
    return record


def cells(cfg, stage):
    if stage not in STAGES:
        raise ValueError(f'Tuning stage must be one of {STAGES}.')
    if stage in ('grid', 'grid_gelu'):
        activation = 'tanh' if stage == 'grid' else 'gelu'
        return [{'label': f'L{layers}_N{width}', 'layers': layers, 'width': width,
                 'fields': {'branch_hidden': [width]*layers, 'trunk_hidden': [width]*layers,
                            'latent_dim': 128, 'deeponet_activation': activation, 'lr': cfg.lr}}
                for layers in cfg.tuning_layers for width in cfg.tuning_neurons]
    grids = [selection(cfg, name) for name in ('grid', 'grid_gelu')]
    if grids[0]['training_settings'] != grids[1]['training_settings'] or grids[0]['seeds'] != grids[1]['seeds']:
        raise ValueError('Tanh and GELU grids must use identical training budgets/settings and seeds.')
    if grids[0]['fields']['lr'] != grids[1]['fields']['lr']:
        raise ValueError('Tanh and GELU grids must use the same initial learning rate.')
    best = min(grids, key=lambda result: result['val_relative_l2'])
    return [{'label': f'lr{rate:.0e}', 'fields': {**best['fields'], 'lr': rate}}
            for rate in cfg.tuning_learning_rates]


def cell_settings(cfg, stage, cell, seed):
    fields = dict(cell['fields'])
    for key in ('branch_hidden', 'trunk_hidden'):
        fields[key] = tuple(fields[key])
    return replace(cfg, **fields, train_size=cfg.tuning_size, training_seed=int(seed),
                   run_name=f'tune_deeponet_N{cfg.tuning_size:04d}_{stage}_{cell["label"]}_seed{seed}',
                   resume=False)


def check_existing(cfg):
    """A finished run can be skipped only when its data/settings still match."""
    path = Path(cfg.log_outdir) / 'training_summary.json'
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding='utf-8'))
    if summary['dataset_sha256'] != file_digest(cfg.dataset_path):
        raise ValueError('Existing run uses different data; choose a new run name.')
    for name in (*FAIR_SETTINGS, 'lr', 'training_seed', 'deeponet_activation', 'latent_dim'):
        if summary['settings'][name] != getattr(cfg, name):
            raise ValueError(f'Existing run setting differs: {name}. Choose a new run name.')
    for name in ('branch_hidden', 'trunk_hidden'):
        if list(summary['settings'][name]) != list(getattr(cfg, name)):
            raise ValueError(f'Existing run architecture differs: {name}.')
    if not Path(cfg.model_path).is_file():
        raise FileNotFoundError('Finished run has no saved model.')
    return True


def run_cell(cfg):
    from .trainer import train_model
    from src.analysis.metrics import evaluate
    from src.analysis.benchmark import hardware_report
    finished = check_existing(cfg)
    if not finished:
        cfg.resume = (Path(cfg.checkpoint_outdir) / 'last_checkpoint.pt').exists() or Path(cfg.model_path).exists()
        train_model(cfg)
        (Path(cfg.log_outdir) / 'hardware.json').write_text(json.dumps(hardware_report(), indent=2))
    elif not (Path(cfg.outdir) / 'all_val_cases_grid_summary.json').exists():
        evaluate(cfg, split='val')
    else:
        print(f'Skipping completed run: {cfg.run_name}')


def run_stage(cfg, stage, seeds, index=None):
    stage_cells = cells(cfg, stage)
    if index is not None:
        if index < 0 or index >= len(stage_cells):
            raise ValueError(f'Cell index must lie in 0..{len(stage_cells)-1}.')
        stage_cells = [stage_cells[index]]
    for cell in stage_cells:
        for seed in seeds:
            run_cell(cell_settings(cfg, stage, cell, seed))


def select_stage(cfg, stage, seeds):
    stage_cells, rows, means = cells(cfg, stage), [], []
    shared = None
    for index, cell in enumerate(stage_cells):
        values = []
        for seed in seeds:
            run = cell_settings(cfg, stage, cell, seed)
            summary_path = Path(run.log_outdir) / 'training_summary.json'
            metrics_path = Path(run.outdir) / 'all_val_cases_grid_summary.json'
            if not summary_path.exists() or not metrics_path.exists():
                raise FileNotFoundError(f'Cell not finished: {run.run_name}')
            summary = json.loads(summary_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            check_existing(run)
            if summary['dataset_sha256'] != file_digest(cfg.dataset_path) or metrics['dataset_sha256'] != summary['dataset_sha256']:
                raise ValueError('Tuning cells use different data.')
            if metrics['checkpoint_sha256'] != file_digest(run.model_path):
                raise ValueError('Validation metrics are stale for a tuning checkpoint.')
            settings = {name: summary['settings'][name] for name in FAIR_SETTINGS}
            if shared is not None and settings != shared:
                raise ValueError('Tuning cells have different budgets/training settings.')
            shared = settings
            expected = cell['fields']
            for name, value in expected.items():
                actual = summary['settings'][name]
                if actual != value and not (isinstance(value, (list, tuple)) and list(actual) == list(value)):
                    raise ValueError(f'Tuning cell field mismatch: {name}')
            row = {'cell': index, 'label': cell['label'], 'seed': seed,
                   'parameters': summary['parameters'], 'best_epoch': summary['best_epoch'],
                   'val_relative_l2': metrics['mean_case_relative_l2'],
                   'val_rmse_K': metrics['global_point_weighted_rmse_K'], 'val_r2': metrics['global_r2']}
            rows.append(row)
            values.append(row)
        means.append({'cell': index, 'label': cell['label'], 'fields': cell['fields'],
                      **{key: float(np.mean([r[key] for r in values])) for key in ('val_relative_l2', 'val_rmse_K', 'val_r2')},
                      'val_relative_l2_sd': float(np.std([r['val_relative_l2'] for r in values], ddof=1)) if len(values)>1 else None})
    best = min(means, key=lambda row: row['val_relative_l2'])
    record = {**best, 'stage': stage, 'seeds': list(seeds), 'train_size_per_family': cfg.tuning_size,
              'dataset_sha256': file_digest(cfg.dataset_path), 'training_settings': shared,
              'criterion': 'lowest mean validation mean-case relative L2 (kelvin)',
              'test_used_for_selection': False, 'cells': means}
    folder = tuning_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    write_rows(folder / f'deeponet_{stage}_cells.csv', rows)
    winner_path(cfg, stage).write_text(json.dumps(record, indent=2), encoding='utf-8')
    from src.analysis.report import plot_tuning
    plot_tuning(cfg, stage, record)
    print(json.dumps({k: v for k,v in record.items() if k != 'cells'}, indent=2))
    return record


def final_settings(cfg, seed, size=None, scaling=False):
    selected = selection(cfg, 'lr')
    fields = dict(selected['fields'])
    for name in ('branch_hidden', 'trunk_hidden'):
        fields[name] = tuple(fields[name])
    size = int(size or cfg.train_size or max(cfg.scaling_sizes))
    name = f'scale_deeponet_N{size:04d}_seed{seed}' if scaling else f'final_deeponet_N{size:04d}_seed{seed}'
    return replace(cfg, **fields, train_size=size, training_seed=int(seed), run_name=name)
