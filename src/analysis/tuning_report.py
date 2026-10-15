"""Read completed activation grids and cache one validation-selected model's fields.

No training or FE solves are performed. Figure rendering is a separate CPU phase.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.data.design import OOD_DESIGN, assert_unique_geometries, file_digest

SCHEMA = 1
ERROR_METRICS = ('rmse_K', 'mae_K', 'relative_l2', 'relative_l2_rise', 'absolute_peak_error_K')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No report rows to write: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_directory(cfg, stage, seeds, model_seed, directory=None):
    if stage not in ('grid', 'grid_gelu'):
        raise ValueError('Detailed grid reports support grid (Tanh) and grid_gelu (GELU).')
    if directory is not None:
        from src.config import PROJECT_ROOT
        path = Path(directory).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    name = 'tanh_grid' if stage == 'grid' else 'gelu_grid'
    return (cfg.output_root / 'reports' / cfg.case_tag / name /
            f'N{cfg.tuning_size:04d}_per_family' /
            ('seeds_' + '_'.join(map(str, seeds))) / f'model_seed{model_seed}')


def representatives(rows):
    """Stable lowest, lower-median and highest relative-L2 cases, within a family."""
    if not rows:
        raise ValueError('Cannot rank an empty family.')
    ordered = sorted(rows, key=lambda row: (row['relative_l2'], row['case_id']))
    return dict(zip(('best', 'median', 'worst'),
                    (ordered[0], ordered[(len(ordered)-1)//2], ordered[-1])))


def family_statistics(rows, domain, family):
    result = {'domain': domain, 'family': family, 'cases': len(rows)}
    for name in ERROR_METRICS:
        values = np.asarray([row[name] for row in rows], dtype=float)
        for label, value in (('mean', values.mean()), ('median', np.median(values)),
                             ('p95', np.percentile(values, 95)), ('max', values.max())):
            result[f'{name}_{label}'] = float(value)
    return result


def _grid_data(cfg, stage, seeds, master):
    from src.training.tuning import cells, cell_settings, select_stage
    # This is the existing validation-only winner, with its existing safeguards.
    selected = select_stage(cfg, stage, seeds, make_plots=False)
    runs, fingerprints = [], {}
    expected_ids = set(map(int, master['val_idx']))
    for index, cell in enumerate(cells(cfg, stage)):
        for seed in seeds:
            run = cell_settings(cfg, stage, cell, seed)
            summary_path = Path(run.log_outdir) / 'training_summary.json'
            metrics_path = Path(run.outdir) / 'all_val_cases_grid_metrics.csv'
            history_path = Path(run.log_outdir) / 'loss_history.csv'
            timing_path = Path(run.log_outdir) / 'epoch_timing.csv'
            summary = read_json(summary_path)
            metrics = read_csv(metrics_path)
            ids = [int(row['case_id']) for row in metrics]
            if len(ids) != len(expected_ids) or set(ids) != expected_ids:
                raise ValueError(f'Validation metrics must cover the complete split: {metrics_path}')
            for row in metrics:
                case_id = int(row['case_id'])
                if (row['geometry_key'] != str(master['geometry_keys'][case_id]) or
                        row['family'] != str(master['geometry_family'][case_id])):
                    raise ValueError(f'Validation case provenance mismatch: {metrics_path}')
            points = sum(int(row['points']) for row in metrics)
            if points <= 0:
                raise ValueError('Validation metrics contain no solid grid points.')
            mean_l2 = float(np.mean([float(row['relative_l2']) for row in metrics]))
            rmse = float(np.sqrt(sum(float(row['squared_error']) for row in metrics) / points))
            original = read_json(Path(run.outdir) / 'all_val_cases_grid_summary.json')
            if not np.allclose([mean_l2, rmse],
                               [original['mean_case_relative_l2'], original['global_point_weighted_rmse_K']],
                               rtol=1e-7, atol=1e-10):
                raise ValueError(f'Validation case table disagrees with its summary: {metrics_path}')
            times = read_csv(timing_path) if timing_path.exists() else []
            full_timing = summary.get('timing_covers_full_history', False)
            seconds = float(times[-1]['elapsed_training_s']) if times and full_timing else None
            runs.append({'cell': index, 'label': cell['label'], 'seed': int(seed),
                         'layers': cell['layers'], 'neurons': cell['width'],
                         'parameters': int(summary['parameters']), 'best_epoch': summary['best_epoch'],
                         'epochs_completed': summary['epochs_completed'],
                         'validation_normalized_mse': float(summary['best_val_loss']),
                         'validation_rmse_K': rmse,
                         'validation_mae_K': sum(float(row['mae_K'])*int(row['points']) for row in metrics)/points,
                         'validation_mean_relative_l2': mean_l2,
                         'training_validation_seconds': seconds,
                         'timing_covers_full_history': bool(full_timing)})
            paths = [summary_path, metrics_path, history_path,
                     Path(run.outdir)/'all_val_cases_grid_summary.json', Path(run.model_path)]
            if timing_path.exists():
                paths.append(timing_path)
            fingerprints[run.run_name] = {path.name: file_digest(path) for path in paths}
    means = []
    for index, cell in enumerate(cells(cfg, stage)):
        group = [row for row in runs if row['cell'] == index]
        row = {'cell': index, 'label': cell['label'], 'layers': cell['layers'], 'neurons': cell['width'],
               'selected': index == selected['cell'], 'parameters': group[0]['parameters']}
        for key in ('validation_normalized_mse', 'validation_rmse_K', 'validation_mae_K',
                    'validation_mean_relative_l2', 'training_validation_seconds'):
            values = [run[key] for run in group]
            row[key] = float(np.mean(values)) if all(v is not None for v in values) else None
        means.append(row)
    return selected, runs, means, fingerprints


def _ood_sources(cfg, master):
    seen = set(master['geometry_keys'].tolist())
    sources = []
    for family in cfg.ood_evaluation_families:
        path = Path(cfg.ood_dataset_path(family, cfg.ood_evaluation_source_sizes[family]))
        if not path.is_file():
            raise FileNotFoundError(f'Missing OOD dataset: {path}. Run generate-ood with the same config first.')
        with np.load(path, allow_pickle=True) as saved:
            data = dict(saved)
        if (str(data['design'][0]) != OOD_DESIGN or str(data['geometry_group'][0]) != 'ood'
                or bool(data['training_eligible'][0])):
            raise ValueError(f'Expected a separate OOD source: {path}')
        assert_unique_geometries(data['geometry_keys'], data['roles'], data['metadata_json'])
        if seen.intersection(data['geometry_keys'].tolist()):
            raise ValueError(f'OOD geometry overlaps another dataset: {family}')
        seen.update(data['geometry_keys'].tolist())
        for key in ('Nx', 'Ny', 'Lx', 'Ly', 'kappa', 'h_hole', 'T_inf', 'q_top'):
            if not np.allclose(data[key], master[key], rtol=0, atol=1e-8):
                raise ValueError(f'OOD physical/grid settings differ: {family}/{key}')
        if not np.array_equal(data['coords'], master['coords']):
            raise ValueError(f'OOD coordinates differ from the model grid: {family}')
        sources.append((family, path, data))
    if not sources:
        raise ValueError('The complete report requires at least one OOD evaluation family.')
    return sources


def _predict_family(cfg, model, coordinates, normalization, data, indices, family, domain, device):
    import torch
    from src.training.runtime import predict_tensor
    from src.analysis.metrics import case_metrics
    predictions = []
    with torch.inference_mode():
        for first in range(0, len(indices), cfg.batch_size):
            ids = indices[first:first+cfg.batch_size]
            raw = torch.as_tensor(data['sdf'][ids], dtype=torch.float32, device=device)
            pred = predict_tensor(model, raw, coordinates, normalization, cfg.point_chunk).cpu().numpy()
            predictions.append(pred*(normalization['T_max']-normalization['T_min']) + normalization['T_min'])
    predictions = np.concatenate(predictions)
    truth = np.asarray(data['temp_physical'][indices], dtype=np.float32)
    mask = np.asarray(data['mask'][indices] > .5)
    rows = []
    for number, (case_id, true, pred, solid) in enumerate(zip(indices, truth, predictions, mask), 1):
        if not solid.any() or not np.isfinite(true[solid]).all() or not np.isfinite(pred[solid]).all():
            raise ValueError(f'Nonfinite or empty solid field: {domain}/{family}/{case_id}')
        metadata = json.loads(str(data['metadata_json'][case_id]))
        if metadata['family'] != family:
            raise ValueError('Case metadata family disagrees with its dataset.')
        true_peak, pred_peak = float(true[solid].max()), float(pred[solid].max())
        row = {'domain': domain, 'family': family, 'family_case_number': number, 'case_id': int(case_id),
               'geometry_key': str(data['geometry_keys'][case_id]),
               'cx_m': metadata['cx'], 'cy_m': metadata['cy'], 'rotation_rad': metadata['rotation'],
               'parameters_json': json.dumps(metadata['parameters'], sort_keys=True),
               **case_metrics(true[solid], pred[solid], float(data['T_inf'][0])),
               'peak_true_K': true_peak, 'peak_pred_K': pred_peak,
               'peak_error_K': pred_peak-true_peak, 'absolute_peak_error_K': abs(pred_peak-true_peak)}
        row['normalized_mse'] = (row['rmse_K'] / (normalization['T_max']-normalization['T_min']))**2
        rows.append(row)
    fields = {'truth_K': np.where(mask, truth, np.nan), 'prediction_K': np.where(mask, predictions, np.nan),
              'mask': mask, 'case_ids': np.asarray(indices),
              'metadata_json': np.asarray(data['metadata_json'][indices], dtype=str),
              'geometry_keys': np.asarray(data['geometry_keys'][indices], dtype=str)}
    for key in ('Nx', 'Ny', 'Lx', 'Ly', 'coords', 'T_inf'):
        fields[key] = data[key]
    return rows, fields


def validate_cache(root, manifest):
    if manifest.get('schema') != SCHEMA or manifest.get('evaluation_complete') is not True:
        raise ValueError('Report cache is incomplete. Run report-tuning --report-phase evaluate first.')
    for name, digest in manifest['cached_files'].items():
        path = Path(root) / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f'Report cache is missing or changed: {path}. Use a new --report-dir.')


def evaluate_report(cfg, stage, seeds, model_seed, root):
    import torch
    from src.data.dataset import load_bundle
    from src.training.tuning import cells, cell_settings
    from src.training.runtime import configure_device, load_checkpoint
    master = load_bundle(cfg)
    selected, runs, means, fingerprints = _grid_data(cfg, stage, seeds, master)
    run = cell_settings(cfg, stage, cells(cfg, stage)[selected['cell']], model_seed)
    ood = _ood_sources(cfg, master)
    provenance = {'schema': SCHEMA, 'stage': stage, 'seeds': list(seeds), 'model_seed': model_seed,
                  'selected_cell': selected['cell'], 'dataset_sha256': selected['dataset_sha256'],
                  'checkpoint_sha256': file_digest(run.model_path), 'grid_inputs': fingerprints,
                  'ood_sha256': {family: file_digest(path) for family, path, data in ood}}
    manifest_path = root / 'report_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get('provenance') != provenance:
            raise ValueError('Report inputs changed. Choose a new --report-dir to keep reports separate.')
        validate_cache(root, old)
        print(f'Reusing verified report predictions: {root}', flush=True)
        return old
    root.mkdir(parents=True, exist_ok=True)
    print(f'Report winner: {selected["label"]}, seed {model_seed}; validation relative L2 only.', flush=True)
    device = configure_device(run)
    model, checkpoint, coordinates = load_checkpoint(run, master, device)
    all_rows, groups, cached = [], [], []
    for domain in ('in_distribution', 'ood'):
        if domain == 'in_distribution':
            test_ids = master['test_idx']
            sources = [(family, master, test_ids[master['geometry_family'][test_ids] == family])
                       for family in sorted(set(master['geometry_family'][test_ids].tolist()))]
        else:
            sources = [(family, data, np.arange(len(data['sdf']))) for family, path, data in ood]
        for family, data, ids in sources:
            print(f'Report inference: {domain}/{family}, {len(ids)} cases', flush=True)
            rows, fields = _predict_family(run, model, coordinates, checkpoint['normalization'],
                                           data, ids, family, domain, device)
            filename = f'data/{domain}/{family}_predictions.npz'
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **fields)
            temporary.replace(path)
            cached.append(filename)
            ranked = representatives(rows)
            groups.append({'domain': domain, 'family': family, 'cases': len(rows), 'fields': filename,
                           'representatives': {label: row['case_id'] for label, row in ranked.items()}})
            all_rows.extend(rows)
    write_csv(root/'tables/grid_runs.csv', runs)
    write_csv(root/'tables/grid_mean_metrics.csv', means)
    write_csv(root/'tables/case_metrics.csv', all_rows)
    family_rows = [family_statistics([row for row in all_rows if row['domain'] == g['domain']
                                     and row['family'] == g['family']], g['domain'], g['family']) for g in groups]
    write_csv(root/'tables/family_summary.csv', family_rows)
    overall = []
    for domain in ('in_distribution', 'ood'):
        rows = [row for row in all_rows if row['domain'] == domain]
        total_points = sum(row['points'] for row in rows)
        row = family_statistics(rows, domain, 'all_families_pooled_cases')
        row.update(solid_points=total_points,
                   pooled_rmse_K=float(np.sqrt(sum(r['squared_error'] for r in rows)/total_points)),
                   pooled_mae_K=sum(r['mae_K']*r['points'] for r in rows)/total_points,
                   normalized_mse=sum(r['normalized_mse']*r['points'] for r in rows)/total_points)
        overall.append(row)
    write_csv(root/'tables/domain_summary.csv', overall)
    for name in ('loss_history.csv', 'epoch_timing.csv'):
        path = Path(run.log_outdir) / name
        if path.exists():
            (root/'tables'/f'selected_{name}').write_bytes(path.read_bytes())
    for path in (root/'tables').glob('*.csv'):
        cached.append(path.relative_to(root).as_posix())
    manifest = {'schema': SCHEMA, 'evaluation_complete': True, 'figures_complete': False,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
                'stage': stage, 'activation': 'tanh' if stage == 'grid' else 'gelu',
                'tuning_size_per_family': cfg.tuning_size, 'seeds': list(seeds), 'model_seed': model_seed,
                'selected': selected, 'selected_run': run.run_name, 'best_epoch': checkpoint['best_epoch'],
                'checkpoint_path': str(Path(run.model_path).resolve()),
                'architecture': checkpoint['architecture'], 'normalization': checkpoint['normalization'],
                'inference_device': str(device), 'torch_version': str(torch.__version__),
                'layers': list(cfg.tuning_layers), 'neurons': list(cfg.tuning_neurons), 'groups': groups,
                'selection_uses_test_or_ood': False,
                'ranking': 'Within each family: relative L2 of Kelvin temperatures; ties by case ID; lower median.',
                'small_family_note': 'Best/median/worst roles may coincide when fewer than three cases exist.',
                'relative_error_map': '100*abs(prediction_K-truth_K)/max(abs(truth_K),1e-12), solid points only',
                'relative_l2_rise': 'norm(prediction_K-truth_K)/norm(truth_K-T_inf), solid points only',
                'test_history_available': False,
                'cached_files': {name: file_digest(root/name) for name in cached}}
    write_json(manifest_path, manifest)
    print(f'Report predictions and tables saved: {root}', flush=True)
    return manifest


def tuning_report(cfg, stage='grid', seeds=(123,), *, phase='all', model_seed=None, directory=None):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('Report tuning seeds must be distinct nonnegative integers.')
    model_seed = int(seeds[0] if model_seed is None else model_seed)
    if model_seed not in seeds:
        raise ValueError('The report model seed must be among the completed tuning seeds.')
    root = report_directory(cfg, stage, seeds, model_seed, directory)
    if phase not in ('all', 'evaluate', 'figures'):
        raise ValueError('Report phase must be all, evaluate or figures.')
    if phase in ('all', 'evaluate'):
        evaluate_report(cfg, stage, seeds, model_seed, root)
    if phase in ('all', 'figures'):
        from src.analysis.tuning_figures import render_report
        render_report(cfg, root)
    return str(root)
