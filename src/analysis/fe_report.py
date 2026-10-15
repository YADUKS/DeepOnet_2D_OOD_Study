"""Optional direct FE-vertex queries, stored separately from the original grid report.

No NGSolve/Netgen imports or FE solves. Torch is imported only for new inference.
The extension can be added to an existing verified Cartesian cache without
rewriting its predictions, metrics or rankings.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from src.data.design import file_digest, geometry_key, resolve_reference
from src.fem.geometry_spec import signed_distance_to_polygon
from .tuning_report import read_json, read_csv, write_json, write_csv, family_statistics, validate_cache

FE_SCHEMA = 1


def mesh_arrays(path, expected_key, Lx, Ly):
    """Validate the saved case identity, field and original triangle connectivity."""
    with np.load(path, allow_pickle=True) as saved:
        needed = ('points', 'triangles', 'temperature', 'boundary_points', 'geometry_record_json')
        if any(name not in saved for name in needed):
            raise ValueError(f'FE cache lacks mesh/temperature/geometry fields: {path}')
        points = np.asarray(saved['points'], dtype=float)
        raw_triangles = np.asarray(saved['triangles'])
        truth = np.asarray(saved['temperature'], dtype=float)
        boundary = np.asarray(saved['boundary_points'], dtype=float)
        metadata = str(saved['geometry_record_json'][0])
    if geometry_key(metadata) != expected_key:
        raise ValueError(f'FE cache belongs to a different geometry: {path}')
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or truth.shape != (len(points),) or boundary.ndim != 2 or boundary.shape[1] != 2
            or not np.isfinite(points).all() or not np.isfinite(truth).all() or not np.isfinite(boundary).all()
            or raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or not len(raw_triangles)
            or not np.issubdtype(raw_triangles.dtype, np.integer)):
        raise ValueError(f'Invalid FE mesh arrays: {path}')
    triangles = raw_triangles.astype(np.int64, copy=True)
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError(f'FE triangle index outside vertex array: {path}')
    edges = points[triangles[:, 1:]]-points[triangles[:, :1]]
    cross = edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0]
    if np.any(np.abs(cross) <= np.finfo(float).eps*max(Lx, Ly)**2):
        raise ValueError(f'Degenerate FE triangle: {path}')
    reverse = cross < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    tolerance = max(Lx, Ly)*2e-6  # saved mesh/boundary coordinates are float32
    if (points[:, 0].min() < -tolerance or points[:, 1].min() < -tolerance
            or points[:, 0].max() > Lx+tolerance or points[:, 1].max() > Ly+tolerance):
        raise ValueError(f'FE vertices outside plate: {path}')
    local = signed_distance_to_polygon(points[:, 0], points[:, 1], boundary)
    if np.any(local < -tolerance):
        raise ValueError(f'FE vertices lie inside the cavity: {path}')
    local[np.abs(local) <= tolerance] = 0.
    return points, triangles, truth, boundary, metadata, local


def validate_fe_cache(root, core, manifest):
    if (manifest.get('schema') != FE_SCHEMA or manifest.get('evaluation_complete') is not True
            or manifest.get('core_provenance') != core['provenance']):
        raise ValueError('FE cache is incomplete or belongs to different report inputs. Run the evaluate phase.')
    for name, digest in manifest['cached_files'].items():
        path = Path(root)/name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f'FE report cache is missing or changed: {path}. Use a new --report-dir.')


def field_path(domain, family, case_id):
    return f'data/fe_vertices/{domain}/{family}/case_{int(case_id):06d}.npz'


def evaluate_fe_report(cfg, root, core, master, ood, *, model=None, checkpoint=None, device=None):
    """Evaluate every ID/OOD case at its own FE vertices, preserving grid rankings."""
    validate_cache(root, core)
    source_lookup = {('in_distribution', g['family']): (Path(cfg.dataset_path), master)
                     for g in core['groups'] if g['domain'] == 'in_distribution'}
    source_lookup.update({('ood', family): (path, data) for family, path, data in ood})
    cases, source_hashes = [], {}
    grid_rows = read_csv(root/'tables/case_metrics.csv')
    for row in grid_rows:
        domain, family, case_id = row['domain'], row['family'], int(row['case_id'])
        owner, data = source_lookup[(domain, family)]
        path = Path(resolve_reference(owner, data['fe_paths'][case_id]))
        if not path.is_file():
            raise FileNotFoundError(f'Missing FE cache: {path}. Restore it or disable fe_vertices in report_field_methods.')
        key = f'{domain}/{family}/{case_id}'
        source_hashes[key] = file_digest(path)
        cases.append((row, data, path))
    manifest_path = root/'fe_vertices_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        validate_fe_cache(root, core, old)
        if old['source_fe_sha256'] != source_hashes:
            raise ValueError('Source FE meshes changed. Use a new --report-dir.')
        print(f'Reusing verified FE-vertex predictions: {root}', flush=True)
        return old
    import torch
    from src.training.runtime import configure_device, load_checkpoint, predict_query_points, predict_tensor, normalize_trunk_coordinates
    from src.training.tuning import cells, cell_settings
    from src.analysis.metrics import case_metrics
    run = cell_settings(cfg, core['stage'], cells(cfg, core['stage'])[core['selected']['cell']], core['model_seed'])
    if model is None:
        device = configure_device(run)
        model, checkpoint, _ = load_checkpoint(run, master, device)
    normalization = checkpoint['normalization']
    rows, comparisons, files, sources, query_check = [], [], [], [], None
    for number, (grid, data, path) in enumerate(cases, 1):
        case_id, family, domain = int(grid['case_id']), grid['family'], grid['domain']
        points, triangles, truth, boundary, metadata, local = mesh_arrays(
            path, grid['geometry_key'], float(data['Lx'][0]), float(data['Ly'][0]))
        branch = torch.as_tensor(data['sdf'][case_id:case_id+1], dtype=torch.float32, device=device)
        if query_check is None:
            # Verify the new helper at identical points with identical inputs.
            xy = torch.as_tensor(normalize_trunk_coordinates(data['coords'], checkpoint['grid']['Lx'], checkpoint['grid']['Ly']), device=device)
            old_prediction = predict_tensor(model, branch, xy, normalization, cfg.point_chunk)
            new_prediction = predict_query_points(model, branch, xy, branch, normalization, cfg.point_chunk)
            difference = float((new_prediction-old_prediction).abs().max().cpu())
            torch.testing.assert_close(new_prediction, old_prediction, rtol=2e-5, atol=2e-6)
            query_check = {'passed': True, 'max_absolute_normalized_difference': difference,
                           'points': len(xy), 'rtol': 2e-5, 'atol': 2e-6}
        saved_length = max(checkpoint['settings']['Lx'], checkpoint['settings']['Ly'])
        length = float(data.get('sdf_normalization_length', [saved_length])[0])
        xy = torch.as_tensor(normalize_trunk_coordinates(points, checkpoint['grid']['Lx'], checkpoint['grid']['Ly']), device=device)
        local_sdf = torch.as_tensor((local/length)[None], dtype=torch.float32, device=device)
        normalized = predict_query_points(model, branch, xy, local_sdf, normalization, cfg.point_chunk)
        prediction = normalized[0].cpu().numpy().astype(float)*(normalization['T_max']-normalization['T_min'])+normalization['T_min']
        if not np.isfinite(prediction).all():
            raise ValueError(f'Nonfinite FE predictions: {domain}/{family}/{case_id}')
        metrics = case_metrics(truth, prediction, float(data['T_inf'][0]))
        row = {'domain': domain, 'family': family, 'family_case_number': int(grid['family_case_number']),
               'case_id': case_id, 'geometry_key': grid['geometry_key'], **metrics,
               'peak_true_K': float(truth.max()), 'peak_pred_K': float(prediction.max()),
               'peak_error_K': float(prediction.max()-truth.max()),
               'absolute_peak_error_K': float(abs(prediction.max()-truth.max()))}
        rows.append(row)
        comparison = {name: row[name] for name in ('domain', 'family', 'case_id', 'family_case_number', 'geometry_key')}
        comparison.update(cartesian_points=int(grid['points']), fe_vertices=len(points))
        for metric in ('rmse_K', 'mae_K', 'relative_l2', 'relative_l2_rise', 'absolute_peak_error_K'):
            comparison[f'cartesian_{metric}'] = float(grid[metric])
            comparison[f'fe_vertices_{metric}'] = row[metric]
        comparisons.append(comparison)
        name = field_path(domain, family, case_id)
        destination = root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, points=points, triangles=triangles, truth_K=truth, prediction_K=prediction,
                            local_sdf_m=local, boundary_points=boundary, metadata_json=np.array([metadata]),
                            geometry_key=np.array([grid['geometry_key']]), case_id=np.array([case_id]))
        temporary.replace(destination)
        files.append(name)
        sources.append({'domain': domain, 'family': family, 'case_id': case_id, 'fields': name})
        if number == 1 or number % 10 == 0 or number == len(cases):
            print(f'FE-vertex inference {number}/{len(cases)}: {domain}/{family}, case {case_id}, {len(points)} vertices', flush=True)
    write_csv(root/'tables/fe_vertices_case_metrics.csv', rows)
    write_csv(root/'tables/case_method_comparison.csv', comparisons)
    summaries = [family_statistics([r for r in rows if r['domain'] == g['domain'] and r['family'] == g['family']],
                                   g['domain'], g['family']) for g in core['groups']]
    write_csv(root/'tables/fe_vertices_family_summary.csv', summaries)
    files += ['tables/fe_vertices_case_metrics.csv', 'tables/case_method_comparison.csv', 'tables/fe_vertices_family_summary.csv']
    manifest = {'schema': FE_SCHEMA, 'evaluation_complete': True, 'method': 'direct_fe_vertices',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'torch_version': str(torch.__version__),
                'numpy_version': str(np.__version__), 'inference_device': str(device), 'point_chunk': cfg.point_chunk,
                'core_provenance': core['provenance'], 'source_fe_sha256': source_hashes,
                'cases': sources, 'query_equivalence_check': query_check,
                'metrics_scope': 'Equal vertex weights; different sampling from Cartesian metrics, not area-weighted FE norms.',
                'ranking': 'Original Cartesian relative-L2 representatives; no FE-based reselection.',
                'reference': 'Saved FE vertex temperatures; contour interpolation is linear inside saved triangles.',
                'cached_files': {name: file_digest(root/name) for name in files}}
    write_json(manifest_path, manifest)
    print(f'FE-vertex evaluation complete: {manifest_path}', flush=True)
    return manifest
