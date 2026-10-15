"""Completed-grid reporting: correct holdouts, one model, ranking and portable cache."""
import json
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


def synthetic_source(cfg, family, ood=False):
    """Analytic fields for testing the report, not FE solutions or research results."""
    count = 3 if ood else 11
    path = Path(cfg.ood_dataset_path(family, 3) if ood else cfg.geometry_dataset_path(family, 5))
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.linspace(0, cfg.Lx, cfg.Nx), np.linspace(0, cfg.Ly, cfg.Ny))
    coords = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    records, sdfs, temperatures = [], [], []
    for i in range(count):
        cx, cy = .012+i*.0007, .01
        parameters = {'outer_radius': .004, 'inner_ratio': .5, 'points': 5} if ood else {'diameter': .006}
        case = GeometryCase(family, cx, cy, parameters=parameters, group='ood' if ood else 'training')
        records.append(json.dumps(case.to_record()))
        angle = np.arctan2(y-cy, x-cx)
        radius = .003 + (.001*np.cos(5*angle) if ood else 0)
        sdf = np.hypot(x-cx, y-cy)-radius
        sdfs.append(sdf.ravel())
        temperature = 310+80*(y/cfg.Ly)+20*np.sin(np.pi*x/cfg.Lx)+i*.5
        temperatures.append(np.where(sdf >= 0, temperature, 0).ravel())
    sdf = np.asarray(sdfs, dtype=np.float32)
    values = dict(schema_version=[2], design=[OOD_DESIGN if ood else DESIGN], geometry_family=[family],
                  geometry_group=['ood' if ood else 'training'], training_eligible=[not ood], n_cases=[count],
                  sdf=sdf, mask=(sdf >= 0).astype(np.float32), temp_physical=np.asarray(temperatures, dtype=np.float32),
                  metadata_json=records, geometry_keys=[geometry_key(record) for record in records],
                  roles=['ood']*count if ood else ['val']*3+['test']*3+['train']*5,
                  candidate_ids=np.arange(count), case_ids=np.arange(count), coords=coords,
                  fe_paths=['fe_cases/case.npz']*count)
    for name in ('Lx', 'Ly', 'Nx', 'Ny', 'maxh', 'order', 'kappa', 'h_hole', 'T_inf', 'q_top'):
        values[name] = [getattr(cfg, name)]
    np.savez(path, **values)
    (path.parent/'fe_cases').mkdir(exist_ok=True)
    np.savez(path.parent/'fe_cases/case.npz', fixture=[True])
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
            self.assertEqual(len(list((self.report/row['folder']).glob('*.png'))), 4)

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


if __name__ == '__main__':
    unittest.main()
