"""Scientific invariants: disjoint geometry, nested prefixes, frozen scale and resume."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.config import Settings
from src.data.compose import compose_training_geometry_dataset
from src.data.dataset import load_bundle
from src.data.design import geometry_key, assert_unique_geometries, design_id
from src.fem.geometry_spec import GeometryCase
from src.training.trainer import train_model
from src.training.tuning import cells, final_settings, run_cell
from src.training.runtime import train_epoch, normalize_trunk_coordinates
from src.models.build import build_model
from src.analysis.metrics import select_evaluation_indices
from tests.test_protocol import source_dataset


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.cfg = Settings(data_root=str(root/'data'), results_root=str(root/'outputs'), Nx=2, Ny=2,
                            training_geometry_counts={'circle': 5}, training_geometry_source_sizes={'circle': 5},
                            validation_cases_per_family={'circle': 1}, test_cases_per_family={'circle': 1},
                            train_size=2, device='cpu', cpu_threads=1, epochs=2, checkpoint_every=1,
                            branch_hidden=(5,), trunk_hidden=(5,), latent_dim=4, run_name='continuous')
        path = source_dataset(self.cfg)
        with np.load(path) as saved:
            values = dict(saved)
        # The maximum training pool has an extreme outside the small prefix.
        values['temp_physical'][-1, 0] = 500
        np.savez(path, **values)
        compose_training_geometry_dataset(self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_canonical_identity_ignores_bookkeeping_and_symmetries(self):
        case = GeometryCase('circle', .01, .01, parameters={'diameter': .01})
        same = replace(case, rotation=1.234, case_id=99, seed=888)
        self.assertEqual(geometry_key(case.to_record()), geometry_key(same.to_record()))
        rect = GeometryCase('rectangle', .01, .01, parameters={'width': .012, 'height': .008})
        rotated = replace(rect, rotation=np.pi)
        self.assertEqual(geometry_key(rect.to_record()), geometry_key(rotated.to_record()))
        metadata = [json.dumps(case.to_record()), json.dumps(same.to_record())]
        with self.assertRaisesRegex(ValueError, 'Duplicate geometry'):
            assert_unique_geometries([geometry_key(x) for x in metadata], ['train','test'], metadata)

    def test_duplicate_detected_when_loading_even_with_distinct_row_indices(self):
        with np.load(self.cfg.dataset_path, allow_pickle=True) as saved:
            data = dict(saved)
        test_id, train_id = data['test_idx'][0], data['train_idx'][0]
        data['geometry_keys'][test_id] = data['geometry_keys'][train_id]
        data['metadata_json'][test_id] = data['metadata_json'][train_id]
        np.savez(self.cfg.dataset_path, **data)
        with self.assertRaisesRegex(ValueError, 'Duplicate geometry'):
            load_bundle(self.cfg)

    def test_nested_prefixes_fixed_holdouts_and_full_pool_temperature_scale(self):
        small = load_bundle(replace(self.cfg, train_size=2))
        large = load_bundle(replace(self.cfg, train_size=4))
        np.testing.assert_array_equal(small['train_idx'], large['train_idx'][:2])
        for key in ('val_idx', 'test_idx', 'T_min', 'T_max'):
            np.testing.assert_array_equal(small[key], large[key])
        self.assertEqual(float(small['T_max'][0]), 500)
        self.assertNotIn(int(small['pool_train_idx'][-1]), small['train_idx'])
        self.assertGreater(float(small['temp'][small['val_idx']].max()), 1)

    def test_loading_data_preserves_generation_and_ood_design_identity(self):
        before = [design_id(self.cfg, 'circle'), design_id(self.cfg, 'star', ood=True)]
        load_bundle(self.cfg)
        self.assertEqual(before, [design_id(self.cfg, 'circle'), design_id(self.cfg, 'star', ood=True)])

    def test_checkpoint_resume_matches_uninterrupted_cpu_training(self):
        train_model(self.cfg, score_validation=False)
        first = replace(self.cfg, epochs=1, run_name='resumed')
        train_model(first, score_validation=False)
        resumed = replace(first, epochs=2, resume=True)
        train_model(resumed, score_validation=False)
        a = torch.load(Path(self.cfg.checkpoint_outdir)/'last_checkpoint.pt', weights_only=True)
        b = torch.load(Path(resumed.checkpoint_outdir)/'last_checkpoint.pt', weights_only=True)
        self.assertEqual(a['history'], b['history'])
        for key in a['model_state']:
            torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
        bad = replace(resumed, lr=.01)
        with self.assertRaisesRegex(ValueError, 'Resume setting mismatch'):
            train_model(bad, score_validation=False)

    def test_chunked_gradient_matches_full_grid(self):
        cfg = replace(self.cfg, point_chunk=4, grad_clip=0)
        torch.manual_seed(91)
        full = build_model(cfg, 2, 2)
        import copy
        chunked = copy.deepcopy(full)
        sdf = torch.rand(2, 4)
        mask = torch.tensor([[1.,0,1,1],[1,1,0,1]])
        target = torch.rand(2,4)
        loader = [(sdf, mask, target)]
        coordinates = torch.rand(4,2)
        norm = {'sdf_mean': 0., 'sdf_std': 1.}
        optimizers = [torch.optim.SGD(model.parameters(), lr=.01) for model in (full, chunked)]
        train_epoch(full, loader, coordinates, optimizers[0], cfg, norm, torch.device('cpu'))
        train_epoch(chunked, loader, coordinates, optimizers[1], replace(cfg, point_chunk=1), norm, torch.device('cpu'))
        for a,b in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(a,b, rtol=1e-6, atol=1e-7)

    def test_reference_grids_and_required_selections(self):
        self.assertEqual(len(cells(self.cfg, 'grid')), 25)
        self.assertEqual(len(cells(self.cfg, 'grid_gelu')), 25)
        self.assertTrue(all(c['fields']['latent_dim'] == 128 for c in cells(self.cfg, 'grid')))
        with self.assertRaises(FileNotFoundError):
            cells(self.cfg, 'lr')
        with self.assertRaises(FileNotFoundError):
            final_settings(self.cfg, 321)

    def test_activation_comparison_rejects_different_training_budgets(self):
        a = {'training_settings': {'epochs': 2}, 'seeds': [123], 'fields': {'lr': .001}}
        b = {**a, 'training_settings': {'epochs': 3}}
        with patch('src.training.tuning.selection', side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, 'identical training budgets'):
                cells(self.cfg, 'lr')

    def test_evaluation_case_numbers_refer_to_each_family_holdout(self):
        data = {'geometry_family': np.array(['circle', 'rectangle', 'circle', 'rectangle', 'circle'])}
        cfg = replace(self.cfg, composed_evaluation_families=('circle',), composed_test_case_numbers={'circle': (1,)})
        np.testing.assert_array_equal(select_evaluation_indices(cfg, data, np.array([1, 2, 3, 4])), [4])
        with self.assertRaisesRegex(ValueError, 'outside'):
            select_evaluation_indices(replace(cfg, composed_test_case_numbers={'circle': (2,)}), data, np.array([1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main()
