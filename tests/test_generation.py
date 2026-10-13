"""Generation rejects overlap before solving and reuses completed FE work."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.config import Settings
from src.data.generation import generate_family
from src.fem.geometry_spec import GeometryCase

class GenerationTests(unittest.TestCase):
    def test_overlap_is_skipped_before_solve_and_extension_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Settings(data_root=str(Path(directory)/'data'), Nx=2, Ny=2,
                           validation_cases_per_family={'circle': 1}, test_cases_per_family={'circle': 1})
            def planner(cfg, family, quota, role):
                centers = {'val': [.01], 'test': [.02], 'train': [.01, .012, .014, .016]}[role]
                return [GeometryCase('circle', x, .01, parameters={'diameter': .008}, candidate_id=i)
                        for i,x in enumerate(centers)]
            calls = []
            def solve(cfg, case):
                calls.append(case.cx)
                sdf = np.array([[1., -1.], [1., 1.]])
                mask = (sdf>0).astype(float)
                temperature = np.array([[310., np.nan], [320.,330.]])
                points = np.array([[0.,0.],[.036,0.],[.036,.021]])
                triangles = np.array([[0,1,2]])
                diagnostics = {'area': .0001, 'perimeter': .04, 'domain_clearance': .001}
                return None,None,sdf,mask,temperature,points,triangles,np.array([310.,320.,330.]),points,diagnostics
            first = generate_family(cfg, 'circle', cfg.geometry_dataset_path('circle', 2),
                                    {'val':1,'test':1,'train':2}, planner, solve)
            self.assertEqual(calls, [.01,.02,.012,.014])
            calls.clear()
            second = generate_family(cfg, 'circle', cfg.geometry_dataset_path('circle', 3),
                                     {'val':1,'test':1,'train':3}, planner, solve)
            self.assertEqual(calls, [.016])
            with np.load(first) as a, np.load(second) as b:
                np.testing.assert_array_equal(a['geometry_keys'], b['geometry_keys'][:len(a['geometry_keys'])])
                self.assertEqual(len(set(b['geometry_keys'])), 5)
            calls.clear()
            generate_family(cfg, 'circle', second, {'val':1,'test':1,'train':3}, planner, solve)
            self.assertEqual(calls, [])

if __name__ == '__main__':
    unittest.main()
