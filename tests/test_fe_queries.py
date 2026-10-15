"""Arbitrary query counts, original-grid equivalence and trusted FE mesh inputs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from src.config import Settings
from src.models.deeponet import DeepONet
from src.training.runtime import predict_query_points, predict_tensor
from src.analysis.fe_report import mesh_arrays
from src.data.design import geometry_key
from src.fem.geometry_spec import GeometryCase


class QueryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.model = DeepONet(12, (7,), (6,), 5)
        self.branch = torch.rand(2, 12)
        self.xy = torch.rand(12, 2)
        self.norm = {'sdf_mean': .13, 'sdf_std': .27}

    def test_identical_grid_inputs_reproduce_original_inference(self):
        expected = predict_tensor(self.model, self.branch, self.xy, self.norm, 5)
        actual = predict_query_points(self.model, self.branch, self.xy, self.branch, self.norm, 5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_arbitrary_fewer_and_more_queries_preserve_weights_and_chunking(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        for count in (1, 7, 31):
            xy, sdf = torch.rand(count, 2), torch.rand(2, count)
            full = predict_query_points(self.model, self.branch, xy, sdf, self.norm, count)
            chunks = predict_query_points(self.model, self.branch, xy, sdf, self.norm, 3)
            self.assertEqual(full.shape, (2, count))
            torch.testing.assert_close(full, chunks, rtol=1e-5, atol=1e-7)
            single = predict_query_points(self.model, self.branch[:1], xy, sdf[:1], self.norm, 3)
            torch.testing.assert_close(single, full[:1], rtol=1e-5, atol=1e-7)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_bad_query_shapes_rejected(self):
        for xy, sdf, chunk in [(torch.rand(5, 3), torch.rand(2, 5), 3),
                               (torch.rand(5, 2), torch.rand(2, 4), 3),
                               (torch.empty(0, 2), torch.empty(2, 0), 3),
                               (self.xy, self.branch, 0)]:
            with self.assertRaises(ValueError):
                predict_query_points(self.model, self.branch, xy, sdf, self.norm, chunk)

    def test_field_method_switch_validation(self):
        from src.data.compose import get_composition_id
        from src.data.design import design_id
        for methods in ([], ['cartesian'], ['fe_vertices'], ['cartesian', 'fe_vertices']):
            cfg = Settings(report_field_methods=methods)
            self.assertEqual(cfg.report_field_methods, tuple(methods))
            self.assertEqual(get_composition_id(cfg), get_composition_id(Settings()))
            self.assertEqual(design_id(cfg, 'star', ood=True), design_id(Settings(), 'star', ood=True))
        for methods in ('both', None, ['other'], ['cartesian', 'cartesian'], [[]], [7]):
            with self.subTest(methods=methods), self.assertRaisesRegex(ValueError, 'report_field_methods'):
                Settings(report_field_methods=methods)

    def test_mesh_identity_connectivity_and_finite_fields_checked(self):
        from tests.test_tuning_report import synthetic_mesh
        cfg = Settings()
        case = GeometryCase('circle', .018, .0105, parameters={'diameter': .008})
        points, triangles, boundary = synthetic_mesh(cfg, case)
        values = dict(points=points, triangles=triangles, boundary_points=boundary,
                      temperature=np.ones(len(points))*350., geometry_record_json=[json.dumps(case.to_record())])
        key = geometry_key(case.to_record())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mesh.npz'
            np.savez(path, **values)
            result = mesh_arrays(path, key, cfg.Lx, cfg.Ly)
            self.assertTrue(np.all(result[-1] >= 0))
            other = replace(case, cx=.017)
            with self.assertRaisesRegex(ValueError, 'different geometry'):
                mesh_arrays(path, geometry_key(other.to_record()), cfg.Lx, cfg.Ly)
            bad = dict(values, temperature=np.full(len(points), np.nan))
            np.savez(path, **bad)
            with self.assertRaisesRegex(ValueError, 'Invalid FE mesh'):
                mesh_arrays(path, key, cfg.Lx, cfg.Ly)
            bad = dict(values, triangles=triangles.copy())
            bad['triangles'][0, 0] = len(points)
            np.savez(path, **bad)
            with self.assertRaisesRegex(ValueError, 'triangle index'):
                mesh_arrays(path, key, cfg.Lx, cfg.Ly)


if __name__ == '__main__':
    unittest.main()
