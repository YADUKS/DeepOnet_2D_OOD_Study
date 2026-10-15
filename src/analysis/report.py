"""DeepONet tuning, seed variability, per-family and data-size reports."""
import csv
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from .metrics import save_figure, write_rows
from src.data.design import file_digest

COMPARISON_SETTINGS = ('epochs', 'batch_size', 'lr', 'precision', 'point_chunk',
                       'grad_clip', 'early_stop_patience', 'scheduler_patience', 'scheduler_factor')


def training_comparison_settings(training):
    return {name: training['settings'][name] for name in COMPARISON_SETTINGS}


def report_dir(cfg):
    return cfg.output_root / 'figures' / cfg.case_tag / 'study_reports'


def plot_tuning(cfg, stage, result):
    import matplotlib.pyplot as plt
    from .output_names import stage_name
    cells = result['cells']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    metrics = (('val_relative_l2', 'Validation relative L2 (kelvin)', 'viridis_r'),
               ('val_rmse_K', 'Validation RMSE (K)', 'viridis_r'),
               ('val_r2', 'Validation R²', 'viridis'))
    for ax, (metric, label, colour) in zip(axes, metrics):
        values = np.array([cell[metric] for cell in cells], dtype=float)
        if stage in ('grid', 'grid_gelu'):
            values = values.reshape(len(cfg.tuning_layers), len(cfg.tuning_neurons))
            im = ax.imshow(values, origin='lower', cmap=colour, aspect='auto')
            ax.set_xticks(range(len(cfg.tuning_neurons)), cfg.tuning_neurons)
            ax.set_yticks(range(len(cfg.tuning_layers)), cfg.tuning_layers)
            ax.set(xlabel='Neurons per hidden layer', ylabel='Hidden layers in branch and trunk')
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    ax.text(col, row, f'{values[row,col]:.2e}', ha='center', va='center', fontsize=8,
                            bbox=dict(facecolor='white', alpha=.6, edgecolor='none'))
            selected_row, selected_col = divmod(result['cell'], len(cfg.tuning_neurons))
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((selected_col-.5, selected_row-.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
            fig.colorbar(im, ax=ax, label=label)
        else:
            rates = [cell['fields']['lr'] for cell in cells]
            ax.plot(rates, values, 'o-')
            ax.axvline(rates[result['cell']], color='red', linestyle=':', label='Selected by relative L2')
            ax.set_xscale('log')
            ax.set(xlabel='Initial learning rate', ylabel=label)
        ax.set_title(label)
    fig.suptitle(f'{stage}: {cfg.tuning_size} training cases per family; red marks validation-relative-L2 winner')
    save_figure(cfg, fig, report_dir(cfg) / stage_name(stage) / f'train{cfg.tuning_size}_per_family' / f'tuning_{stage}')


def collect_runs(cfg, size, seeds, prefix='scale'):
    records = []
    shared = None
    for seed in seeds:
        run = replace(cfg, train_size=size, run_name=f'{prefix}_deeponet_N{size:04d}_seed{seed}')
        summary = json.loads((Path(run.outdir)/'all_test_cases_grid_summary.json').read_text())
        training = json.loads((Path(run.log_outdir)/'training_summary.json').read_text())
        if summary['checkpoint_sha256'] != file_digest(run.model_path):
            raise ValueError('Stale test metrics: checkpoint changed.')
        if summary['dataset_sha256'] != file_digest(run.dataset_path):
            raise ValueError('Report dataset changed.')
        if training['dataset_sha256'] != summary['dataset_sha256'] or summary['seed'] != seed or training['seed'] != seed:
            raise ValueError('Report training and evaluation provenance do not match.')
        if summary['train_size_per_family'] != size or training['train_size_per_family'] != size:
            raise ValueError('Report run has a different training size.')
        signature = (summary['dataset_sha256'], summary['architecture'], summary['train_size_per_family'],
                     training_comparison_settings(training))
        if shared is not None and signature != shared:
            raise ValueError('Seed runs use different datasets, architectures, sizes or training settings.')
        shared = signature
        with (Path(run.outdir)/'all_test_cases_grid_metrics.csv').open() as stream:
            cases = list(csv.DictReader(stream))
        case_ids = [row['geometry_key'] for row in cases]
        if records and case_ids != records[0]['case_ids']:
            raise ValueError('Seed runs were evaluated on different test cases.')
        records.append({'seed': seed, 'summary': summary, 'training': training, 'cases': cases,
                        'case_ids': case_ids, 'cfg': run})
    return records


def summarize(cfg, size, seeds, prefix='scale'):
    import matplotlib.pyplot as plt
    records = collect_runs(cfg, size, seeds, prefix)
    folder = report_dir(cfg)/'seed_variability_study'/f'train{size}_per_family'
    rows = [{'seed': rec['seed'], 'train_size_per_family': size, 'total_training_cases': rec['training']['train_cases'],
             **{key: rec['summary'][key] for key in ('mean_case_relative_l2','mean_case_rmse_K','global_point_weighted_rmse_K','global_r2')}}
            for rec in records]
    write_rows(folder/'seeds.csv', rows)
    result = {'seeds': list(seeds), 'train_size_per_family': size,
              'total_training_cases': records[0]['training']['train_cases'],
              'scope': 'Mean and sample SD across model seeds on identical held-out geometries.'}
    for key in ('mean_case_relative_l2', 'mean_case_rmse_K', 'global_point_weighted_rmse_K', 'global_r2'):
        values = [row[key] for row in rows]
        result[key] = {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values)>1 else None}
    (folder/'seed_summary.json').write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for rec in records:
        history = np.atleast_1d(np.genfromtxt(Path(rec['cfg'].log_outdir)/'loss_history.csv', delimiter=',', names=True))
        axes[0].semilogy(history['epoch'], history['val_point_weighted_mse'], label=str(rec['seed']))
    axes[0].set(xlabel='Epoch', ylabel='Validation masked MSE')
    axes[0].legend(title='Training seed')
    axes[1].plot([str(row['seed']) for row in rows], [row['mean_case_rmse_K'] for row in rows], 'o')
    axes[1].set(xlabel='Training seed', ylabel='Test mean case RMSE (K)')
    save_figure(cfg, fig, folder/'seed_variability')
    family_rows = []
    families = sorted(records[0]['summary']['by_family'])
    for family in families:
        values = [rec['summary']['by_family'][family]['mean_case_rmse_K'] for rec in records]
        family_rows.append({'family': family, 'mean_case_rmse_K': float(np.mean(values)),
                            'seed_sd_K': float(np.std(values, ddof=1)) if len(values)>1 else 0.0})
    write_rows(folder/'errors_by_family.csv', family_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].bar(families, [row['mean_case_rmse_K'] for row in family_rows], yerr=[row['seed_sd_K'] for row in family_rows], capsize=3)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('Test mean case RMSE (K)')
    matrix = np.array([[float(row['rmse_K']) for row in rec['cases']] for rec in records])
    axes[1].hist(matrix.mean(axis=0), bins=min(20, matrix.shape[1]))
    axes[1].set(xlabel='Case RMSE averaged across seeds (K)', ylabel='Test cases')
    save_figure(cfg, fig, folder/'test_errors')
    return result


def scaling_report(cfg, sizes, seeds):
    import matplotlib.pyplot as plt
    rows, signatures = [], []
    for size in sizes:
        records = collect_runs(cfg, size, seeds)
        signatures.append((records[0]['summary']['dataset_sha256'], records[0]['summary']['architecture'], records[0]['case_ids'],
                           training_comparison_settings(records[0]['training'])))
        for record in records:
            rows.append({'train_size_per_family': size, 'total_training_cases': record['training']['train_cases'],
                         'seed': record['seed'], 'relative_l2': record['summary']['mean_case_relative_l2'],
                         'rmse_K': record['summary']['mean_case_rmse_K']})
    if any(value != signatures[0] for value in signatures):
        raise ValueError('Scaling runs must use identical architecture, training settings, data and held-out cases.')
    folder = report_dir(cfg)/'dataset_size_study'
    write_rows(folder/'scaling.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    rates = {}
    for ax, metric, label in zip(axes, ('relative_l2', 'rmse_K'), ('Test relative L2 (kelvin)', 'Test mean case RMSE (K)')):
        groups = [[row[metric] for row in rows if row['train_size_per_family']==size] for size in sizes]
        means = np.array([np.mean(values) for values in groups])
        sd = np.array([np.std(values, ddof=1) if len(values)>1 else 0 for values in groups])
        ax.errorbar(sizes, means, yerr=sd, fmt='o', capsize=4, label='Mean ± seed SD')
        if len(sizes)>1 and np.all(means>0):
            slope, intercept = np.polyfit(np.log(sizes), np.log(means), 1)
            rates[metric] = {'alpha': float(-slope), 'C': float(np.exp(intercept))}
            ax.plot(sizes, np.exp(intercept)*np.array(sizes)**slope, label=f'Fit: alpha = {-slope:.3f}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.set(xlabel='Training cases per family', ylabel=label)
        ax.legend()
    save_figure(cfg, fig, folder/'scaling')
    output = {'sizes_per_family': list(sizes), 'seeds': list(seeds), 'rates': rates,
              'fit': 'Least squares of log(seed-mean error) on log(cases per family).',
              'dataset_sha256': signatures[0][0]}
    (folder/'scaling_rates.json').write_text(json.dumps(output, indent=2))
    return output
