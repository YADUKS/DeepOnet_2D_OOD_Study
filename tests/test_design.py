import unittest
import numpy as np
from src.config import Settings
from src.data.design import halton_sequence, latin_hypercube_sampling, role_splits, design_id
from src.data.generate import build_simple_family_case_plan
from src.data.external import build_ood_family_case_plan
from src.fem.geometry_spec import TRAINING_GEOMETRY_FAMILIES, OOD_GEOMETRY_FAMILIES, transform_points, validate_boundary
from src.fem.simple_geometries import build_canonical_simple_boundary
from src.fem.ood_geometries import build_canonical_ood_boundary


class DesignTests(unittest.TestCase):
    def test_halton_known_values_and_prefix(self):
        np.testing.assert_allclose(halton_sequence(3, 2), [[.5, 1/3], [.25, 2/3], [.75, 1/9]])
        np.testing.assert_array_equal(halton_sequence(8, 8), halton_sequence(50, 8)[:8])
        points = halton_sequence(50, 8)
        self.assertTrue(np.all((points >= 0) & (points < 1)))

    def test_lhs_strata_reproducibility(self):
        samples = latin_hypercube_sampling(19, 8, 12)
        np.testing.assert_array_equal(samples, latin_hypercube_sampling(19, 8, 12))
        for column in samples.T:
            np.testing.assert_array_equal(np.sort((column * 19).astype(int)), np.arange(19))

    def test_family_prefixes_and_fixed_holdouts(self):
        cfg = Settings()
        for family in TRAINING_GEOMETRY_FAMILIES:
            small = build_simple_family_case_plan(cfg, family, 5)
            large = build_simple_family_case_plan(cfg, family, 17)
            self.assertGreaterEqual(len(small), 5, family)
            self.assertEqual([c.to_record() for c in small[:5]], [c.to_record() for c in large[:5]])
            heldout = build_simple_family_case_plan(cfg, family, 4, "val")
            cfg.geometry_cases_per_family[family] = 10000
            self.assertEqual([c.to_record() for c in heldout],
                             [c.to_record() for c in build_simple_family_case_plan(cfg, family, 4, "val")])
            self.assertNotEqual(heldout[0].to_record(), build_simple_family_case_plan(cfg, family, 4, "test")[0].to_record())

    def test_all_family_boundaries(self):
        cfg = Settings()
        for families, planner, builder in ((TRAINING_GEOMETRY_FAMILIES, build_simple_family_case_plan, build_canonical_simple_boundary),
                                            (OOD_GEOMETRY_FAMILIES, build_ood_family_case_plan, build_canonical_ood_boundary)):
            for family in families:
                valid = 0
                for case in planner(cfg, family, 3):
                    boundary = transform_points(builder(case, curve_segments=cfg.curved_boundary_segments),
                                                case.cx, case.cy, case.rotation, case.scale)
                    try:
                        validate_boundary(boundary, cfg.Lx, cfg.Ly, cfg.geometry_minimum_clearance,
                                          cfg.geometry_minimum_area, cfg.geometry_minimum_edge_length)
                        valid += 1
                    except ValueError:
                        pass
                    if valid == 3:
                        break
                self.assertEqual(valid, 3, family)

    def test_invalid_roles_rejected(self):
        with self.assertRaises(ValueError):
            role_splits(["train", "val", "ood"], ["circle"] * 3)

    def test_unrelated_family_settings_do_not_change_source_identity(self):
        cfg = Settings()
        circle = design_id(cfg, "circle")
        cfg.star_outer_radius_max *= 0.9
        cfg.ellipse_aspect_ratio_max *= 0.9
        self.assertEqual(circle, design_id(cfg, "circle"))
        cfg.D_max *= 0.9
        self.assertNotEqual(circle, design_id(cfg, "circle"))


if __name__ == "__main__":
    unittest.main()
