"""Standalone publication figures from a verified report cache; no Torch or FE imports."""
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .tuning_report import read_csv, read_json, write_csv, write_json, validate_cache

METRICS = (
    ('rmse_K', 'RMSE [K]', 1),
    ('mae_K', 'MAE [K]', 1),
    ('relative_l2', 'Relative L2 [%]', 100),
    ('relative_l2_rise', 'Temperature-rise relative L2 [%]', 100),
    ('absolute_peak_error_K', 'Absolute peak-temperature error [K]', 1),
)
RANK_STYLE = {'best': ('#12805c', 'o'), 'median': ('#bb7610', 's'), 'worst': ('#b92536', '^')}


def numeric_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key, value in row.items():
            if key in ('domain', 'family', 'geometry_key', 'parameters_json', 'label'):
                continue
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value == '':
                row[key] = None
            else:
                row[key] = float(value)
    return rows


class Figures:
    def __init__(self, cfg, root, manifest):
        self.cfg, self.root, self.manifest = cfg, root, manifest
        self.index = []
        self.model = f'{manifest["activation"].upper()} {manifest["selected"]["label"]}, seed {manifest["model_seed"]}'

    def save(self, figure, filename, caption):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        try:
            for extension in (('png', 'pdf') if self.cfg.save_pdf_figures else ('png',)):
                target = path.with_suffix('.'+extension)
                figure.savefig(target, dpi=self.cfg.publish_dpi, bbox_inches='tight')
                files.append(target.relative_to(self.root).as_posix())
        finally:
            plt.close(figure)
        self.index.append({'name': path.name, 'caption': caption, 'files': files})
        print(f'Figure {len(self.index)}: {files[0]}', flush=True)


def grid_maps(output, rows):
    manifest = output.manifest
    layers, neurons = manifest['layers'], manifest['neurons']
    lookup = {(int(r['layers']), int(r['neurons'])): r for r in rows}
    if len(lookup) != len(layers)*len(neurons):
        raise ValueError('Grid report must contain every layer/neuron cell exactly once.')
    chosen = next(row for row in rows if row['selected'])
    metrics = (
        ('validation_normalized_mse', 'Validation normalized MSE', 1),
        ('validation_rmse_K', 'Validation RMSE [K]', 1),
        ('validation_mae_K', 'Validation MAE [K]', 1),
        ('validation_mean_relative_l2', 'Mean validation relative L2 [%]', 100),
    )
    for key, label, scale in metrics:
        values = np.asarray([[lookup[(l, n)][key]*scale for n in neurons] for l in layers])
        fig, ax = plt.subplots(figsize=(7.8, 6.4), layout='constrained')
        im = ax.imshow(values, origin='lower', interpolation='nearest', aspect='auto', cmap='viridis_r')
        ax.set(xticks=range(len(neurons)), xticklabels=neurons, yticks=range(len(layers)),
               yticklabels=layers, xlabel='Neurons per hidden layer', ylabel='Hidden layers (branch and trunk)',
               title=f'{label}\n{manifest["activation"].upper()} grid; {manifest["tuning_size_per_family"]} training cases/family')
        for j in range(len(layers)):
            for i in range(len(neurons)):
                r, g, b, _ = im.cmap(im.norm(values[j, i]))
                color = 'black' if .2126*r+.7152*g+.0722*b > .55 else 'white'
                value = values[j, i]
                ax.text(i, j, f'{value:.2e}' if key.endswith('mse') else f'{value:.3g}',
                        ha='center', va='center', color=color, fontsize=10)
        ax.set_xticks(np.arange(len(neurons)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(layers)+1)-.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        x, y = neurons.index(int(chosen['neurons'])), layers.index(int(chosen['layers']))
        ax.add_patch(Rectangle((x-.47, y-.47), .94, .94, fill=False, edgecolor='#ef3340', linewidth=3))
        ax.text(.5, -.18, f'Red outline: validation-selected {chosen["label"]}. Same network on every map.\n'
                f'{len(manifest["seeds"])} tuning seed(s); cells show arithmetic means across seeds.',
                transform=ax.transAxes, ha='center', fontsize=9)
        fig.colorbar(im, ax=ax, label=label, shrink=.85)
        output.save(fig, f'01_grid_maps/{key}', label+'; discrete cells, no interpolation between networks.')


def convergence(output, cases):
    history = numeric_rows(output.root/'tables/selected_loss_history.csv')
    epochs = [row['epoch'] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    ax.semilogy(epochs, [r['train_point_weighted_mse'] for r in history], label='Training', color='#2264a5')
    ax.semilogy(epochs, [r['val_point_weighted_mse'] for r in history], label='Validation', color='#d27b24')
    test = [r for r in cases if r['domain'] == 'in_distribution']
    scale = output.manifest['normalization']['T_max']-output.manifest['normalization']['T_min']
    test_loss = sum(r['squared_error'] for r in test)/sum(r['points'] for r in test)/scale**2
    best_epoch = output.manifest['best_epoch']
    ax.axvline(best_epoch, color='0.45', linestyle=':', label=f'Selected epoch {best_epoch}')
    ax.scatter([best_epoch], [test_loss], color='#12805c', marker='*', s=130, zorder=5,
               label=f'Test at selected checkpoint: {test_loss:.3e}')
    ax.set(xlabel='Epoch', ylabel='Solid-point weighted normalized MSE', title=f'Loss history | {output.model}')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    ax.text(.5, -.17, 'Test score is one checkpoint evaluation. No per-epoch test history was recorded.',
            transform=ax.transAxes, ha='center', fontsize=9)
    output.save(fig, '02_convergence/training_validation_loss', 'Training/validation curves and a single held-out test score.')
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    ax.step(epochs, [r['lr'] for r in history], where='post', color='#2264a5')
    ax.set(xlabel='Epoch', ylabel='Learning rate', yscale='log', title=f'Learning-rate history | {output.model}')
    ax.grid(alpha=.2)
    output.save(fig, '02_convergence/learning_rate_vs_epoch', 'Recorded learning rate, without reconstructing missing epochs.')


def family_errors(output, group, rows, folder):
    rows = sorted(rows, key=lambda row: row['family_case_number'])
    for key, label, scale in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), layout='constrained')
        ax.plot([r['family_case_number'] for r in rows], [r[key]*scale for r in rows],
                '-o', color='#2264a5', markersize=3, linewidth=1)
        for role, case_id in group['representatives'].items():
            row = next(r for r in rows if int(r['case_id']) == case_id)
            color, marker = RANK_STYLE[role]
            ax.scatter([row['family_case_number']], [row[key]*scale], c=color, marker=marker, s=65,
                       label=f'{role.capitalize()} L2 case: {int(row["family_case_number"])} (ID {case_id})', zorder=5)
        ax.set(xlabel='Test scenario within family (original order)', ylabel=label,
               title=f'{group["family"].replace("_", " ").title()} | {group["domain"].replace("_", " ")}\n{output.model}')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        output.save(fig, f'{folder}/errors/{key}_by_case', label+' for every family case; marked cases ranked by relative L2.')


def family_fields(output, group, rows, folder):
    with np.load(output.root/group['fields'], allow_pickle=False) as archive:
        fields = dict(archive)
    shape = (int(fields['Ny'][0]), int(fields['Nx'][0]))
    coords = fields['coords'].reshape(*shape, 2)*1000
    ids = fields['case_ids'].tolist()
    choices = {role: ids.index(case_id) for role, case_id in group['representatives'].items()}
    selected = list(choices.values())
    true, pred = fields['truth_K'], fields['prediction_K']
    absolute = np.abs(pred-true)
    relative = 100*absolute/np.maximum(np.abs(true), 1e-12)
    low = float(min(np.nanmin(true[selected]), np.nanmin(pred[selected])))
    high = float(max(np.nanmax(true[selected]), np.nanmax(pred[selected])))
    if high == low:
        high = low+1e-6
    limits = {'temperature_K': [low, high], 'absolute_error_K': [0., max(float(np.nanmax(absolute[selected])), 1e-12)],
              'relative_error_percent': [0., max(float(np.nanmax(relative[selected])), 1e-12)]}
    selections = []
    for role, index in choices.items():
        case_id = int(fields['case_ids'][index])
        row = next(row for row in rows if int(row['case_id']) == case_id)
        metadata = json.loads(str(fields['metadata_json'][index]))
        case_folder = f'{folder}/contours/{role}_case_{case_id:06d}'
        write_json(output.root/case_folder/'case_details.json', {
            'rank': role, 'case_metrics': row, 'geometry': metadata, 'color_limits': limits,
            'checkpoint_sha256': output.manifest['provenance']['checkpoint_sha256'],
            'field_source': group['fields'], 'grid_field': True})
        selections.append({'domain': group['domain'], 'family': group['family'], 'rank': role,
                           'case_id': case_id, 'family_case_number': int(row['family_case_number']),
                           'relative_l2': row['relative_l2'], 'rmse_K': row['rmse_K'], 'folder': case_folder})
        maps = (
            ('ground_truth_temperature_K', 'Ground-truth temperature', true[index], 'Temperature [K]', limits['temperature_K'], 'inferno'),
            ('prediction_temperature_K', 'Predicted temperature', pred[index], 'Temperature [K]', limits['temperature_K'], 'inferno'),
            ('absolute_error_K', 'Absolute temperature error', absolute[index], 'Absolute error [K]', limits['absolute_error_K'], 'magma'),
            ('relative_error_percent', 'Pointwise relative temperature error', relative[index], 'Relative error [%]', limits['relative_error_percent'], 'magma'),
        )
        for name, title, values, label, bounds, cmap in maps:
            fig, ax = plt.subplots(figsize=(8, 5.7), layout='constrained')
            im = ax.pcolormesh(coords[..., 0], coords[..., 1], np.ma.masked_invalid(values.reshape(shape)),
                               shading='nearest', cmap=cmap, vmin=bounds[0], vmax=bounds[1], rasterized=True)
            ax.set(xlim=(0, float(fields['Lx'][0])*1000), ylim=(0, float(fields['Ly'][0])*1000),
                   xlabel='x [mm]', ylabel='y [mm]', aspect='equal',
                   title=f'{title}\n{group["family"].replace("_", " ").title()} | {role} L2 case {case_id} | {group["domain"].replace("_", " ")}')
            fig.colorbar(im, ax=ax, label=label, shrink=.8)
            ax.text(.5, -.28, f'{output.model}\nRMSE = {row["rmse_K"]:.4g} K; relative L2 = {100*row["relative_l2"]:.4g}%\n'
                    f'Centre = ({metadata["cx"]*1000:.3g}, {metadata["cy"]*1000:.3g}) mm; rotation = {metadata["rotation"]:.3g} rad',
                    transform=ax.transAxes, ha='center', fontsize=9)
            output.save(fig, f'{case_folder}/{name}',
                        f'{title}; {group["domain"]}/{group["family"]}/{role}, case {case_id}. Solid grid points only; cavity masked.')
    return selections


def distributions(output, cases):
    for domain in ('in_distribution', 'ood'):
        families = sorted(set(row['family'] for row in cases if row['domain'] == domain))
        for key, label, scale in METRICS:
            values = [[r[key]*scale for r in cases if r['domain'] == domain and r['family'] == f] for f in families]
            fig, ax = plt.subplots(figsize=(max(7.5, len(families)*1.1), 5.6), layout='constrained')
            ax.boxplot(values, positions=range(len(families)), showmeans=True, patch_artist=True,
                       boxprops={'facecolor': '#c9daee'}, medianprops={'color': '#b92536'}, widths=.55)
            ax.set_xticks(range(len(families)), [f.replace('_', ' ') for f in families], rotation=32, ha='right')
            ax.set(ylabel=label, title=f'{domain.replace("_", " ").title()} family distribution\n{output.model}')
            ax.grid(axis='y', alpha=.2)
            output.save(fig, f'05_additional/family_distributions/{domain}_{key}_boxplot',
                        label+' by family; box=IQR, line=median, triangle=mean, whiskers=1.5 IQR, dots=outliers.')
    for key, label, scale in (METRICS[0], METRICS[2]):
        values = [[r[key]*scale for r in cases if r['domain'] == d] for d in ('in_distribution', 'ood')]
        fig, ax = plt.subplots(figsize=(7, 5.2), layout='constrained')
        ax.boxplot(values, positions=[0, 1], showmeans=True)
        ax.set_xticks([0, 1], ['In-distribution test', 'OOD'])
        ax.set(ylabel=label, title=f'ID / OOD error distribution | {output.model}')
        ax.grid(axis='y', alpha=.2)
        output.save(fig, f'05_additional/generalization/{key}_id_vs_ood',
                    label+' descriptive ID/OOD comparison; cases are pooled, without paired-case claims.')


def cost_plots(output, grid):
    for key, label, filename in (
        ('parameters', 'Trainable scalar parameters', 'accuracy_vs_parameters'),
        ('training_validation_seconds', 'Recorded training + validation time [s]', 'accuracy_vs_training_time'),
    ):
        fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained')
        available = [r for r in grid if r[key] is not None]
        for row in available:
            chosen = row['selected']
            ax.scatter(row[key], 100*row['validation_mean_relative_l2'], c='#b92536' if chosen else '#2264a5',
                       marker='*' if chosen else 'o', s=140 if chosen else 35, zorder=4 if chosen else 2)
            ax.annotate(row['label'], (row[key], 100*row['validation_mean_relative_l2']),
                        xytext=(4, 4), textcoords='offset points', fontsize=7)
        if key == 'parameters' and available:
            ax.set_xscale('log')
        if not available:
            ax.text(.5, .5, 'Complete timing history is unavailable.\nNo training times were inferred.',
                    transform=ax.transAxes, ha='center')
        ax.set(xlabel=label, ylabel='Mean validation relative L2 [%]', title=f'Grid accuracy and cost | {output.manifest["activation"].upper()}')
        ax.grid(alpha=.2)
        ax.text(.5, -.18, f'Red star: validation-selected network. {len(grid)-len(available)} cells omitted for missing data.\n'
                'Time is observed per run; concurrent jobs and hardware can affect the comparison.',
                transform=ax.transAxes, ha='center', fontsize=9)
        output.save(fig, f'05_additional/accuracy_cost/{filename}',
                    'Validation accuracy versus '+label+'. Means across configured seeds; timing excludes incomplete histories.')


def report_readme(output):
    m = output.manifest
    lines = ['# Completed activation-grid report', '', f'Selected checkpoint: **{output.model}**, epoch {m["best_epoch"]}.', '',
             f'Training size: {m["tuning_size_per_family"]} cases per family. Grid seeds: {m["seeds"]}.', '',
             'Architecture selection uses the lowest mean validation case-relative L2 in Kelvin. '
             'The same validation-selected network is outlined on all four discrete grid maps. '
             'Grid cells are not smoothed. With multiple seeds, grid values are arithmetic seed means; '
             'the prediction checkpoint uses the explicitly recorded model seed (default: first configured seed).', '',
             'Test/OOD cases never select an architecture, checkpoint epoch or model seed. '
             'Best/median/worst refer to cases within each family, ordered by that one model\'s relative L2; '
             'ties use case ID and even-sized families use the lower median. Roles may coincide in tiny families.', '',
             '## Metrics and fields', '',
             '- RMSE/MAE grid maps pool solid points within each validation split. Relative L2 is averaged per case.',
             '- Loss is MSE after the saved training-only temperature normalization.',
             '- Relative L2 = ||prediction - truth|| / ||truth||, using Kelvin temperatures.',
             '- Temperature-rise relative L2 = ||prediction - truth|| / ||truth - ambient||; near-zero rise can make this large.',
             '- Pointwise relative error (%) = 100 |prediction - truth| / max(|truth|, 1e-12), at solid points.',
             '- Peak error is predicted maximum minus true maximum; the plotted absolute peak error uses its magnitude.',
             '- Field figures show the saved Cartesian evaluation grid, not newly generated FE solutions. Cavities are masked.',
             '- For each family, ground truth/prediction share a temperature scale across the three selected cases. '
             'Each error type also shares its scale across those cases. Different families have separate scales.',
             '- Training/validation histories are recorded curves. The test score is a single point at the selected epoch; '
             'no test curve is fabricated and no model is retrained.', '',
             '## Tables and cached data', '']
    for path in sorted((output.root/'tables').glob('*.csv')):
        name = path.relative_to(output.root).as_posix()
        lines.append(f'- [{path.name}]({name})')
    lines += ['', 'All evaluated fields and geometry metadata are saved in `data/`. '
              'The manifest records input fingerprints and cache checksums. The figures phase needs only this '
              'report folder plus the plotting code, NumPy and Matplotlib; original datasets, checkpoints and CUDA '
              'are unnecessary. See `figure_index.json` for a machine-readable inventory.', '', '## Individual figures', '']
    for item in output.index:
        links = ' / '.join(f'[{path.rsplit(".", 1)[-1].upper()}]({path})' for path in item['files'])
        lines.append(f'- {item["caption"]} {links}')
    (output.root/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def render_report(cfg, root):
    manifest = read_json(root/'report_manifest.json')
    validate_cache(root, manifest)
    cases = numeric_rows(root/'tables/case_metrics.csv')
    grid = numeric_rows(root/'tables/grid_mean_metrics.csv')
    output = Figures(cfg, root, manifest)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        grid_maps(output, grid)
        convergence(output, cases)
        selected = []
        for group in manifest['groups']:
            base = '03_in_distribution' if group['domain'] == 'in_distribution' else '04_ood'
            folder = f'{base}/{group["family"]}'
            rows = [r for r in cases if r['domain'] == group['domain'] and r['family'] == group['family']]
            family_errors(output, group, rows, folder)
            selected.extend(family_fields(output, group, rows, folder))
        write_csv(root/'tables/representative_cases.csv', selected)
        distributions(output, cases)
        cost_plots(output, grid)
    write_json(root/'figure_index.json', output.index)
    report_readme(output)
    manifest.update(figures_complete=True, figures_created_utc=datetime.now(timezone.utc).isoformat(),
                    figure_count=len(output.index), figure_files=[p for item in output.index for p in item['files']],
                    figure_dpi=cfg.publish_dpi, figures_saved_as_pdf=cfg.save_pdf_figures)
    write_json(root/'report_manifest.json', manifest)
    print(f'Report complete: {len(output.index)} separate figures. Open {root/"README.md"}', flush=True)
    return manifest
