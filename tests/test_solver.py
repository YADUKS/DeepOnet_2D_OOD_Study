"""Run separately from Torch tests to isolate native FE/ML runtimes."""
import unittest
import numpy as np
from src.config import Settings
from src.data.generate import build_simple_family_case_plan
from src.fem.sampling import sample_geometry_case


class SolverTests(unittest.TestCase):
    def test_finite_temperature_and_heat_balance(self):
        from ngsolve import Integrate
        cfg = Settings(Nx=24, Ny=18, maxh=0.0015)
        case = build_simple_family_case_plan(cfg, "circle", 1)[0]
        mesh, gfu, sdf, mask, temperature, points, triangles, temperature_fe, _, _ = sample_geometry_case(cfg, case)
        self.assertTrue(np.isfinite(temperature[mask > .5]).all())
        self.assertTrue(np.isfinite(temperature_fe).all())
        self.assertGreater(temperature_fe.min(), cfg.T_inf)
        heat_out = Integrate(cfg.h_hole * (gfu - cfg.T_inf), mesh, definedon=mesh.Boundaries("hole"))
        heat_in = cfg.q_top * cfg.Lx
        self.assertLess(abs(heat_out - heat_in) / heat_in, 1e-6)


if __name__ == "__main__":
    unittest.main()
