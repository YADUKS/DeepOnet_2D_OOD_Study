"""Completed-grid reporting: correct holdouts, one model, ranking and portable cache."""
import json
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.config import PROJECT_ROOT, Settings
from src.data.compose import compose_training_geometry_dataset
from src.data.design import DESIGN, OOD_DESIGN, file_digest, geometry_key
from src.fem.geometry_spec import GeometryCase
from src.training.tuning import cells, cell_settings, run_stage
from src.analysis.tuning_report import (read_json, read_csv, write_csv, representatives,
                                       tuning_report)


def synthetic_mesh(cfg, case):
    """Conforming polygon-ring fixture, no FE solve or native mesher."""
    from src.fem.geometry_spec import transform_points
    from src.fem.simple_geometries import build_canonical_simple_boundary
    from src.fem.ood_geometries import build_canonical_ood_boundary
    builder = build_canonical_ood_boundary if case.group == 'ood' else build_canonical_simple_boundary
    boundary = transform_points(builder(case, cfg.curved_boundary_segments), case.cx, case.cy, case.rotation, case.scale)
    centre = np.array([case.cx, case.cy])
    corners = np.array([[0., 0.], [cfg.Lx, 0.], [cfg.Lx, cfg.Ly], [0., cfg.Ly]])
    angle_points = np.vstack((boundary, corners))-centre
    angles = np.unique(np.round(np.mod(np.arctan2(angle_points[:, 1], angle_points[:, 0]), 2*np.pi), 12))
    inner, outer = [], []
    edges = np.roll(boundary, -1, axis=0)-boundary
    start = boundary-centre
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        cross = direction[0]*edges[:, 1]-direction[1]*edges[:, 0]
        valid = np.abs(cross) > 1e-15
        denominator = np.where(valid, cross, 1.)
        t = (start[:, 0]*edges[:, 1]-start[:, 1]*edges[:, 0])/denominator
        u = (start[:, 0]*direction[1]-start[:, 1]*direction[0])/denominator
        selected = valid & (t > 0) & (u >= -1e-9) & (u <= 1+1e-9)
        inner.append(centre+t[selected].min()*direction)
        lengths = [((bound if component > 0 else 0)-coordinate)/component if abs(component) > 1e-14 else np.inf
                   for component, coordinate, bound in zip(direction, centre, (cfg.Lx, cfg.Ly))]
        outer.append(centre+min(lengths)*direction)
    inner, outer = np.array(inner), np.array(outer)
    points = np.vstack((inner, (inner+outer)/2, outer))
    n = len(inner)
    triangles = []
    for ring in range(2):
        for i in range(n):
            j = (i+1) % n
            a, b, c, d = ring*n+i, ring*n+j, (ring+1)*n+i, (ring+1)*n+j
            triangles.extend([[a, c, d], [a, d, b]])
    return points.astype(np.float32), np.asarray(triangles, dtype=np.int32), boundary.astype(np.float32)


def synthetic_source(cfg, family, ood=False):
    """Analytic fields for testing the report, not FE solutions or research results."""
    count = 3 if ood else 11
    path = Path(cfg.ood_dataset_path(family, 3) if ood else cfg.geometry_dataset_path(family, 5))
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.linspace(0, cfg.Lx, cfg.Nx), np.linspace(0, cfg.Ly, cfg.Ny))
    coords = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    records, sdfs, temperatures, fe_paths = [], [], [], []
    (path.parent/'fe_cases').mkdir(exist_ok=True)
    for i in range(count):
        cx, cy = .012+i*.0007, .01
        parameters = {'outer_radius': .004, 'inner_ratio': .5, 'points': 5} if ood else {'diameter': .006}
        case = GeometryCase(family, cx, cy, parameters=parameters, case_id=i, group='ood' if ood else 'training')
        records.append(json.dumps(case.to_record()))
        angle = np.arctan2(y-cy, x-cx)
        radius = .003 + (.001*np.cos(5*angle) if ood else 0)
        sdf = np.hypot(x-cx, y-cy)-radius
        sdfs.append(sdf.ravel())
        temperature = 310+80*(y/cfg.Ly)+20*np.sin(np.pi*x/cfg.Lx)+i*.5
        temperatures.append(np.where(sdf >= 0, temperature, 0).ravel())
        points, triangles, boundary = synthetic_mesh(cfg, case)
        fe_truth = 310+80*(points[:, 1]/cfg.Ly)+20*np.sin(np.pi*points[:, 0]/cfg.Lx)+i*.5
        fe_paths.append(f'fe_cases/case_{i:06d}.npz')
        np.savez(path.parent/fe_paths[-1], points=points, triangles=triangles, boundary_points=boundary,
                 temperature=fe_truth, geometry_record_json=[records[-1]])
    sdf = np.asarray(sdfs, dtype=np.float32)
    values = dict(schema_version=[2], design=[OOD_DESIGN if ood else DESIGN], geometry_family=[family],
                  geometry_group=['ood' if ood else 'training'], training_eligible=[not ood], n_cases=[count],
                  sdf=sdf, mask=(sdf >= 0).astype(np.float32), temp_physical=np.asarray(temperatures, dtype=np.float32),
                  metadata_json=records, geometry_keys=[geometry_key(record) for record in records],
                  roles=['ood']*count if ood else ['val']*3+['test']*3+['train']*5,
                  candidate_ids=np.arange(count), case_ids=np.arange(count), coords=coords,
                  fe_paths=fe_paths)
    for name in ('Lx', 'Ly', 'Nx', 'Ny', 'maxh', 'order', 'kappa', 'h_hole', 'T_inf', 'q_top'):
        values[name] = [getattr(cfg, name)]
    np.savez(path, **values)
    return path


class RankingTests(unittest.TestCase):
    def test_lower_median_ties_and_single_case(self):
        rows = [{'case_id': i, 'relative_l2': error} for i, error in [(7, .4), (3, .1), (2, .1), (6, .8)]]
        selected = representatives(rows)
        self.assertEqual([selected[k]['case_id'] for k in ('best', 'median', 'worst')], [2, 3, 6])
        self.assertEqual(len({r['case_id'] for r in representatives(rows[:1]).values()}), 1)
        with self.assertRaises(ValueError):
            representatives([])

    def test_seed_and_stage_validation(self):
        for seeds in ([], [123, 123], [-1]):
            with self.assertRaisesRegex(ValueError, 'seeds'):
                tuning_report(Settings(), seeds=seeds, phase='evaluate')
        with self.assertRaisesRegex(ValueError, 'seed'):
            tuning_report(Settings(), seeds=[123], model_seed=456, phase='evaluate')
        with self.assertRaisesRegex(ValueError, 'grid'):
            tuning_report(Settings(), stage='lr', phase='evaluate')


class CompletedGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cfg = Settings(data_root=str(cls.root/'data'), results_root=str(cls.root/'outputs'), Nx=16, Ny=12,
                           training_geometry_counts={'circle': 5}, training_geometry_source_sizes={'circle': 5},
                           validation_cases_per_family={'circle': 3}, test_cases_per_family={'circle': 3},
                           ood_evaluation_families=('star',), ood_evaluation_source_sizes={'star': 3},
                           ood_cases_per_family={'star': 3},
                           train_size=2, tuning_size=2, tuning_layers=(1, 2), tuning_neurons=(4, 6),
                           device='cpu', cpu_threads=1, epochs=2, tuning_epochs=2, checkpoint_every=1,
                           batch_size=2, point_chunk=96, publish_dpi=55, save_pdf_figures=False,
                           composed_plot_cases_per_family=0)
        synthetic_source(cls.cfg, 'circle')
        synthetic_source(cls.cfg, 'star', ood=True)
        compose_training_geometry_dataset(cls.cfg)
        run_stage(cls.cfg, 'grid', [123])
        cls.report = cls.root/'report'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_missing_grid_cell_cannot_select_partial_winner(self):
        run = cell_settings(self.cfg, 'grid', cells(self.cfg, 'grid')[-1], 123)
        summary = Path(run.log_outdir)/'training_summary.json'
        original = summary.read_bytes()
        summary.unlink()
        try:
            with self.assertRaisesRegex(FileNotFoundError, 'Cell not finished'):
                tuning_report(self.cfg, directory=self.report, phase='evaluate')
            self.assertFalse((self.report/'report_manifest.json').exists())
        finally:
            summary.write_bytes(original)

    def test_02_full_report_one_model_and_solid_only_metrics(self):
        model_paths = [Path(cell_settings(self.cfg, 'grid', c, 123).model_path) for c in cells(self.cfg, 'grid')]
        before = {str(p): file_digest(p) for p in model_paths}
        with patch('src.training.trainer.train_model', side_effect=AssertionError('Report must not retrain')):
            tuning_report(self.cfg, directory=self.report)
        self.assertEqual(before, {str(p): file_digest(p) for p in model_paths})
        manifest = read_json(self.report/'report_manifest.json')
        self.assertTrue(manifest['figures_complete'])
        self.assertFalse(manifest['test_history_available'])
        self.assertFalse(manifest['selection_uses_test_or_ood'])
        self.assertEqual(len(list((self.report/'01_grid_maps').glob('*.png'))), 4)
        rows = read_csv(self.report/'tables/case_metrics.csv')
        self.assertEqual(len(rows), 6)
        for group in manifest['groups']:
            family_rows = [r for r in rows if r['domain'] == group['domain']]
            ranked = representatives([dict(r, relative_l2=float(r['relative_l2']), case_id=int(r['case_id'])) for r in family_rows])
            self.assertEqual(group['representatives'], {k: v['case_id'] for k, v in ranked.items()})
            with np.load(self.report/group['fields']) as fields:
                self.assertTrue(np.isnan(fields['prediction_K'][~fields['mask']]).all())
                for row in family_rows:
                    i = fields['case_ids'].tolist().index(int(row['case_id']))
                    solid = fields['mask'][i]
                    true, pred = fields['truth_K'][i][solid].astype(float), fields['prediction_K'][i][solid].astype(float)
                    error = pred-true
                    self.assertEqual(int(row['points']), int(solid.sum()))
                    self.assertAlmostEqual(float(row['rmse_K']), np.sqrt(np.mean(error**2)), places=7)
                    self.assertAlmostEqual(float(row['relative_l2']), np.linalg.norm(error)/np.linalg.norm(true), places=8)
                    self.assertAlmostEqual(float(row['absolute_peak_error_K']), abs(pred.max()-true.max()), places=7)
                    self.assertAlmostEqual(float(row['relative_l2_rise']), np.linalg.norm(error)/np.linalg.norm(true-self.cfg.T_inf), places=8)
        grid = read_csv(self.report/'tables/grid_mean_metrics.csv')
        expected = min(grid, key=lambda row: float(row['validation_mean_relative_l2']))
        self.assertEqual(manifest['selected']['label'], expected['label'])
        for source in read_csv(self.report/'tables/grid_runs.csv'):
            cell = cells(self.cfg, 'grid')[int(source['cell'])]
            run = cell_settings(self.cfg, 'grid', cell, 123)
            val = read_csv(Path(run.outdir)/'all_val_cases_grid_metrics.csv')
            points = sum(int(row['points']) for row in val)
            expected_mae = sum(float(row['mae_K'])*int(row['points']) for row in val)/points
            self.assertAlmostEqual(float(source['validation_mae_K']), expected_mae, places=9)
        selections = read_csv(self.report/'tables/representative_cases.csv')
        self.assertEqual(len(selections), 6)
        for row in selections:
            directory = self.report/row['folder']/'cartesian'/f'{row["rank"]}_case_{int(row["case_id"]):06d}'
            self.assertEqual(len(list(directory.glob('*.png'))), 4)

    def test_03_cache_reuse_requires_no_inference(self):
        with patch('src.analysis.tuning_report._predict_family', side_effect=AssertionError('Unexpected inference')):
            tuning_report(self.cfg, directory=self.report, phase='evaluate')

    def test_04_cache_change_rejected_before_render(self):
        path = self.report/'tables/case_metrics.csv'
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        try:
            with self.assertRaisesRegex(ValueError, 'changed'):
                tuning_report(self.cfg, directory=self.report, phase='figures')
        finally:
            path.write_bytes(original)

    def test_05_relocated_cache_renders_without_torch_or_original_inputs(self):
        destination = self.root/'portable_report'
        shutil.copytree(self.report, destination)
        # A standalone process refuses Torch/FE imports; original data is hidden.
        script = '''import importlib.abc, sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('torch', 'ngsolve', 'netgen'):
            raise AssertionError('Plotting imported ' + fullname)
sys.meta_path.insert(0, BlockML())
from run import arguments, settings_from_arguments, dispatch
args = arguments()
dispatch(args, settings_from_arguments(args))
'''
        env = dict(os.environ, MPLBACKEND='Agg')
        subprocess.run([sys.executable, '-c', script, 'report-tuning', '--report-phase', 'figures',
                        '--report-dir', str(destination), '--results-root', str(self.root/'missing'),
                        '--data-root', str(self.root/'missing_data'), '--no-pdf', '--dpi', '55'],
                       cwd=PROJECT_ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        self.assertTrue(read_json(destination/'report_manifest.json')['figures_complete'])
        artifact = os.environ.get('REPORT_TEST_ARTIFACT_DIR')
        if artifact:
            target = Path(artifact)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.report, target, dirs_exist_ok=True)

    def test_06_stale_validation_case_table_rejected(self):
        run = cell_settings(self.cfg, 'grid', cells(self.cfg, 'grid')[0], 123)
        path = Path(run.outdir)/'all_val_cases_grid_metrics.csv'
        original = path.read_bytes()
        rows = read_csv(path)
        rows[0]['relative_l2'] = float(rows[0]['relative_l2'])+1
        write_csv(path, rows)
        try:
            with self.assertRaisesRegex(ValueError, 'disagrees with its summary'):
                tuning_report(self.cfg, directory=self.root/'stale_report', phase='evaluate')
        finally:
            path.write_bytes(original)

    def test_07_existing_models_and_cache_survive_output_renaming(self):
        import torch
        from src.analysis.output_names import CATEGORIES, rename_outputs
        from src.analysis.tuning_report import report_directory
        from src.data.dataset import load_bundle
        from src.training.runtime import load_checkpoint
        from src.training.tuning import check_existing, selection
        files = [path for category in CATEGORIES for path in
                 (self.cfg.output_root/category/self.cfg.case_tag).rglob('*') if path.is_file()]
        before = {str(path.relative_to(self.cfg.output_root)): file_digest(path) for path in files}
        legacy_cache = (self.cfg.output_root/'reports'/self.cfg.legacy_case_tag/'tanh_grid'/
                        f'N{self.cfg.tuning_size:04d}_per_family'/'seeds_123'/'model_seed123')
        shutil.copytree(self.report, legacy_cache)
        self.assertEqual(report_directory(self.cfg, 'grid', [123], 123), legacy_cache)
        # Recreate a checkout that still has the original hash-only output roots.
        for category in CATEGORIES:
            source = self.cfg.output_root/category/self.cfg.case_tag
            if source.exists():
                source.rename(source.with_name(self.cfg.legacy_case_tag))
        run = cell_settings(self.cfg, 'grid', cells(self.cfg, 'grid')[0], 123)
        self.assertIn(self.cfg.legacy_case_tag, run.model_path)
        self.assertTrue(check_existing(run))
        self.assertFalse(selection(self.cfg, 'grid')['test_used_for_selection'])
        model, checkpoint, coords = load_checkpoint(run, load_bundle(self.cfg), torch.device('cpu'))
        self.assertEqual(checkpoint['settings']['training_seed'], 123)
        rename_outputs(self.cfg)
        for name, digest in before.items():
            self.assertEqual(file_digest(self.cfg.output_root/name), digest)
        # Renaming parent directories cannot force new inference or invalidate
        # fingerprints of the original model/metrics/history files.
        with patch('src.analysis.tuning_report._predict_family', side_effect=AssertionError('Unexpected inference')):
            result = tuning_report(self.cfg, phase='evaluate')
        expected_cache = self.cfg.output_root/'reports'/self.cfg.case_tag/legacy_cache.relative_to(
            self.cfg.output_root/'reports'/self.cfg.legacy_case_tag)
        self.assertEqual(Path(result), expected_cache)
        self.assertTrue(read_json(expected_cache/'report_manifest.json')['figures_complete'])

    def test_08_fe_cache_adds_queries_without_changing_cartesian_results(self):
        cfg = replace(self.cfg, report_field_methods=('cartesian', 'fe_vertices'))
        before = {name: file_digest(self.report/name) for name in read_json(self.report/'report_manifest.json')['cached_files']}
        with patch('src.analysis.tuning_report._predict_family', side_effect=AssertionError('Unexpected grid inference')):
            tuning_report(cfg, directory=self.report, phase='evaluate')
        self.assertEqual(before, {name: file_digest(self.report/name) for name in before})
        manifest = read_json(self.report/'fe_vertices_manifest.json')
        self.assertTrue(manifest['query_equivalence_check']['passed'])
        rows = read_csv(self.report/'tables/fe_vertices_case_metrics.csv')
        self.assertEqual(len(rows), 6)
        counts = []
        for case, row in zip(manifest['cases'], rows):
            with np.load(self.report/case['fields']) as saved:
                error = saved['prediction_K']-saved['truth_K']
                self.assertAlmostEqual(float(row['rmse_K']), np.sqrt(np.mean(error**2)), places=9)
                self.assertEqual(int(row['points']), len(saved['points']))
                counts.append(len(saved['points']))
        self.assertGreater(len(set(counts)), 1)
        with patch('src.training.runtime.predict_query_points', side_effect=AssertionError('Unexpected FE inference')):
            tuning_report(cfg, directory=self.report, phase='evaluate')

    def test_09_both_method_plots_use_identical_reference_only_scales(self):
        cfg = replace(self.cfg, report_field_methods=('cartesian', 'fe_vertices'))
        tuning_report(cfg, directory=self.report, phase='figures')
        index = read_json(self.report/'figure_index.json')
        self.assertEqual(sum('/contours/cartesian/' in f['files'][0] for f in index), 24)
        self.assertEqual(sum('/contours/fe_vertices/' in f['files'][0] for f in index), 24)
        for g in read_json(self.report/'report_manifest.json')['groups']:
            base = '03_in_distribution' if g['domain'] == 'in_distribution' else '04_ood'
            for rank, case_id in g['representatives'].items():
                parent = self.report/base/g['family']/'contours'
                a = read_json(parent/'cartesian'/f'{rank}_case_{case_id:06d}'/'case_details.json')
                b = read_json(parent/'fe_vertices'/f'{rank}_case_{case_id:06d}'/'case_details.json')
                self.assertEqual(a['color_limits'], b['color_limits'])
                with np.load(self.report/g['fields']) as grid, np.load(self.report/b['field_source']) as fe:
                    i = grid['case_ids'].tolist().index(case_id)
                    truth = np.concatenate((grid['truth_K'][i][grid['mask'][i]], fe['truth_K']))
                    self.assertEqual(a['color_limits']['temperature_K'], [float(truth.min()), float(truth.max())])
        if os.environ.get('REPORT_FE_TEST_ARTIFACT_DIR'):
            shutil.copytree(self.report, os.environ['REPORT_FE_TEST_ARTIFACT_DIR'], dirs_exist_ok=True)

    def test_10_disabling_a_method_skips_its_figures(self):
        for methods in [('fe_vertices',), ()]:
            cfg = replace(self.cfg, report_field_methods=methods)
            tuning_report(cfg, directory=self.report, phase='figures')
            index = read_json(self.report/'figure_index.json')
            self.assertFalse(any('/contours/cartesian/' in f['files'][0] for f in index))
            count = sum('/contours/fe_vertices/' in f['files'][0] for f in index)
            self.assertEqual(count, 24 if methods else 0)

    def test_11_fe_cache_tamper_is_rejected_and_cartesian_only_still_works(self):
        path = self.report/'tables/fe_vertices_case_metrics.csv'
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        try:
            with self.assertRaisesRegex(ValueError, 'FE report cache'):
                tuning_report(replace(self.cfg, report_field_methods=('fe_vertices',)), directory=self.report, phase='figures')
            # No FE file is opened if only the current Cartesian method is requested.
            with patch('src.analysis.fe_report.validate_fe_cache', side_effect=AssertionError('FE disabled')):
                tuning_report(self.cfg, directory=self.report, phase='figures')
        finally:
            path.write_bytes(original)

    def test_12_fresh_report_can_evaluate_both_methods_without_fe_solves(self):
        cfg = replace(self.cfg, report_field_methods=('cartesian', 'fe_vertices'))
        destination = self.root/'fresh_both_methods'
        with patch('src.fem.sampling.sample_geometry_case', side_effect=AssertionError('No FE solve')), \
             patch('src.fem.sampling.sample_ood_geometry_case', side_effect=AssertionError('No OOD FE solve')):
            tuning_report(cfg, directory=destination, phase='evaluate')
        manifest = read_json(destination/'fe_vertices_manifest.json')
        self.assertTrue(manifest['evaluation_complete'])
        self.assertEqual(manifest['inference_device'], 'cpu')
        self.assertIn('torch_version', manifest)

    def test_13_fe_only_cache_renders_offline_without_torch_or_ngsolve(self):
        destination = self.root/'portable_fe_report'
        shutil.copytree(self.report, destination)
        config = self.root/'portable_fe.json'
        config.write_text(json.dumps({'report_field_methods': ['fe_vertices'], 'publish_dpi': 55, 'save_pdf_figures': False}))
        script = '''import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('torch', 'ngsolve', 'netgen'):
            raise AssertionError('Offline plotting imported ' + fullname)
sys.meta_path.insert(0, BlockNative())
from run import arguments, settings_from_arguments, dispatch
args = arguments()
dispatch(args, settings_from_arguments(args))
'''
        subprocess.run([sys.executable, '-c', script, 'report-tuning', '--report-phase', 'figures',
                        '--config', str(config), '--report-dir', str(destination),
                        '--data-root', str(self.root/'missing_sources')], cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
        self.assertEqual(read_json(destination/'report_manifest.json')['rendered_field_methods'], ['fe_vertices'])


if __name__ == '__main__':
    unittest.main()
