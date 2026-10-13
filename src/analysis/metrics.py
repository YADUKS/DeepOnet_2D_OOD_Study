"""Validation-only model selection, held-out testing and separate OOD scoring."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.data.dataset import load_bundle
from src.data.design import assert_unique_geometries, file_digest
from src.training.runtime import configure_device, load_checkpoint, predict_tensor


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_metrics(true, pred, ambient=300.0):
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    error = pred - true
    squared = float(error @ error)
    norm = float(true @ true)
    centered = float(((true-true.mean())**2).sum())
    return {'rmse_K': float(np.sqrt(squared / len(true))),
            'mae_K': float(np.abs(error).mean()), 'max_absolute_error_K': float(np.abs(error).max()),
            'relative_l2': float(np.sqrt(squared / max(norm, 1e-30))),
            'relative_l2_rise': float(np.sqrt(squared / max(float(((true-ambient)**2).sum()), 1e-30))),
            'r2': None if centered == 0 else 1-squared/centered,
            'points': len(true), 'squared_error': squared,
            'true_sum': float(true.sum()), 'true_squared_sum': norm}


def aggregate(rows):
    points = sum(r['points'] for r in rows)
    sse = sum(r['squared_error'] for r in rows)
    sst = sum(r['true_squared_sum'] for r in rows) - sum(r['true_sum'] for r in rows)**2 / points
    return {'cases': len(rows), 'solid_points': points,
            'mean_case_rmse_K': float(np.mean([r['rmse_K'] for r in rows])),
            'global_point_weighted_rmse_K': float(np.sqrt(sse / points)),
            'mean_case_relative_l2': float(np.mean([r['relative_l2'] for r in rows])),
            'mean_case_relative_l2_rise': float(np.mean([r['relative_l2_rise'] for r in rows])),
            'global_r2': None if sst <= 0 else 1-sse/sst}


def save_figure(cfg, figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix('.png'), dpi=cfg.publish_dpi, bbox_inches='tight')
    if cfg.save_pdf_figures:
        figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(figure)


def plot_case(cfg, data, case_id, prediction, folder):
    import matplotlib.pyplot as plt
    shape = (int(data['Ny'][0]), int(data['Nx'][0]))
    mask = data['mask'][case_id].reshape(shape) > .5
    true = data['temp_physical'][case_id].reshape(shape)
    pred = prediction.reshape(shape)
    low, high = min(true[mask].min(), pred[mask].min()), max(true[mask].max(), pred[mask].max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    extent = [0, float(data['Lx'][0])*1000, 0, float(data['Ly'][0])*1000]
    for ax, values, title, cmap in zip(axes, (true, pred, abs(pred-true)),
                                     ('FE temperature', 'DeepONet prediction', 'Absolute error'),
                                     ('viridis', 'viridis', 'magma')):
        kwargs = {'vmin': low, 'vmax': high} if cmap == 'viridis' else {'vmin': 0}
        im = ax.imshow(np.ma.array(values, mask=~mask), origin='lower', extent=extent, cmap=cmap, **kwargs)
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)')
        fig.colorbar(im, ax=ax, label='K', shrink=.8)
    family = str(data['geometry_family'][case_id])
    fig.suptitle(f'{family} — case {case_id}')
    save_figure(cfg, fig, Path(folder) / f'{family}_case_{case_id:06d}')


def select_evaluation_indices(cfg, data, indices, external=False):
    """Case numbers are positions within each family's held-out set."""
    indices = np.asarray(indices, dtype=np.int64)
    if not external and cfg.composed_evaluation_families:
        indices = indices[np.isin(data['geometry_family'][indices], cfg.composed_evaluation_families)]
    requests = cfg.ood_test_case_numbers if external else cfg.composed_test_case_numbers
    if not requests:
        return indices
    selected = []
    for family in dict.fromkeys(data['geometry_family'][indices].tolist()):
        family_ids = indices[data['geometry_family'][indices] == family]
        if family in requests:
            positions = np.asarray(requests[family], dtype=np.int64)
            if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= len(family_ids)):
                raise ValueError(f'Requested case is outside the {family} held-out set.')
            family_ids = family_ids[positions]
        selected.extend(family_ids.tolist())
    return np.asarray(selected, dtype=np.int64)


def evaluate(cfg, split='test', external=False, selected_only=False, make_plots=False):
    if split not in ('val', 'test'):
        raise ValueError('Only validation and held-out test evaluation are supported.')
    device = configure_device(cfg)
    ood_sources = [(family, cfg.ood_dataset_path(family, cfg.ood_evaluation_source_sizes[family]))
                   for family in cfg.ood_evaluation_families] if external else []
    master = load_bundle(cfg)
    model, checkpoint, coordinates = load_checkpoint(cfg, master, device)
    norm = checkpoint['normalization']
    datasets = []
    if external:
        seen = set(master['geometry_keys'].tolist())
        for family, path in ood_sources:
            with np.load(path, allow_pickle=True) as saved:
                data = dict(saved)
            assert_unique_geometries(data['geometry_keys'], data['roles'], data['metadata_json'])
            if seen.intersection(data['geometry_keys'].tolist()):
                raise ValueError(f'OOD geometry overlaps an in-distribution/other OOD set: {family}')
            seen.update(data['geometry_keys'].tolist())
            for key in ('Nx', 'Ny', 'Lx', 'Ly', 'kappa', 'h_hole', 'T_inf', 'q_top'):
                if not np.allclose(data[key], master[key], rtol=0, atol=1e-8):
                    raise ValueError(f'OOD and training physical/grid settings differ: {key}')
            data['geometry_family'] = np.full(len(data['sdf']), family)
            datasets.append((family, data, np.arange(len(data['sdf'])), file_digest(path)))
    else:
        datasets.append((split, master, master[split+'_idx'], checkpoint['dataset_sha256']))
    all_rows, external_hashes = [], {}
    partial = False
    for label, data, indices, fingerprint in datasets:
        original_ids = indices
        if external:
            external_hashes[label] = fingerprint
        indices = select_evaluation_indices(cfg, data, indices, external)
        partial |= set(indices.tolist()) != set(original_ids.tolist())
        if selected_only:
            if cfg.test_case_number < 0 or cfg.test_case_number >= len(indices):
                raise ValueError('Requested case is outside the selected split.')
            indices = indices[cfg.test_case_number:cfg.test_case_number+1]
        plotted = {}
        for first in range(0, len(indices), cfg.batch_size):
            ids = indices[first:first+cfg.batch_size]
            raw = torch.as_tensor(data['sdf'][ids], dtype=torch.float32, device=device)
            prediction = predict_tensor(model, raw, coordinates, norm, cfg.point_chunk).cpu().numpy()
            prediction = prediction * (norm['T_max']-norm['T_min']) + norm['T_min']
            for case_id, pred in zip(ids, prediction):
                family = str(data['geometry_family'][case_id])
                solid = data['mask'][case_id] > .5
                row = {'case_id': int(case_id), 'family': family,
                       'geometry_key': str(data['geometry_keys'][case_id]),
                       **case_metrics(data['temp_physical'][case_id][solid], pred[solid], float(data['T_inf'][0]))}
                all_rows.append(row)
                if make_plots and (selected_only or plotted.get(family, 0) < cfg.composed_plot_cases_per_family):
                    plot_case(cfg, data, case_id, pred, Path(cfg.figure_outdir) / ('ood' if external else split))
                    plotted[family] = plotted.get(family, 0) + 1
    if not all_rows:
        raise ValueError('No evaluation cases selected.')
    summary = {**aggregate(all_rows), 'dataset_sha256': checkpoint['dataset_sha256'],
               'checkpoint_sha256': file_digest(cfg.model_path), 'architecture': checkpoint['architecture'],
               'seed': checkpoint['settings']['training_seed'],
               'train_size_per_family': checkpoint['settings']['train_size'],
               'split': 'ood' if external else split, 'external_sha256': external_hashes,
               'by_family': {family: aggregate([r for r in all_rows if r['family'] == family])
                             for family in sorted({r['family'] for r in all_rows})}}
    # Plotting an individual case must never replace the complete selection metrics.
    if not selected_only:
        scope = 'ood' if external else split
        folder = Path(cfg.outdir)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = 'selected' if partial else 'all'
        write_rows(folder / f'{prefix}_{scope}_cases_grid_metrics.csv', all_rows)
        (folder / f'{prefix}_{scope}_cases_grid_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'{summary["split"]}: {len(all_rows)} cases; mean case RMSE={summary["mean_case_rmse_K"]:.6g} K')
    return summary


def plot_loss(cfg):
    import matplotlib.pyplot as plt
    data = np.genfromtxt(Path(cfg.log_outdir) / 'loss_history.csv', delimiter=',', names=True)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for key, label in [('train_point_weighted_mse', 'Training'), ('val_point_weighted_mse', 'Validation')]:
        ax.semilogy(np.atleast_1d(data['epoch']), np.atleast_1d(data[key]), label=label)
    ax.set(xlabel='Epoch', ylabel='Masked MSE (normalised temperature)')
    ax.legend()
    save_figure(cfg, fig, Path(cfg.figure_outdir) / 'loss_history')
