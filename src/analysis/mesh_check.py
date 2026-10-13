"""Three-mesh FE convergence and boundary heat-balance check for each family."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

def run_mesh_check(cfg):
    from ngsolve import Integrate, ds, CoefficientFunction
    from src.data.generate import build_simple_family_case_plan
    from src.fem.sampling import sample_geometry_case
    rows = []
    for family in cfg.generate_geometry_families:
        case = build_simple_family_case_plan(cfg, family, 1)[0]
        previous = None
        for factor in (2.0, 1.0, .5):
            mesh_cfg = replace(cfg, maxh=cfg.maxh*factor)
            mesh, gfu, sdf, mask, temperature, *_ = sample_geometry_case(mesh_cfg, case)
            incoming = float(Integrate(CoefficientFunction(cfg.q_top) * ds('top'), mesh))
            outgoing = float(Integrate(cfg.h_hole*(gfu-cfg.T_inf)*ds('hole'), mesh))
            valid = mask>.5
            if not valid.any() or not np.isfinite(temperature[valid]).all():
                raise ValueError(f'Invalid solid-grid temperatures in mesh check: {family}')
            difference = None if previous is None else float(np.sqrt(np.mean((temperature[valid]-previous[valid])**2)))
            rows.append({'family': family, 'maxh_m': mesh_cfg.maxh, 'elements': mesh.ne,
                         'mean_temperature_K': float(np.nanmean(temperature[valid])),
                         'grid_rmse_vs_previous_mesh_K': difference,
                         'heat_in_W_per_m': incoming, 'heat_out_W_per_m': outgoing,
                         'relative_heat_imbalance': abs(incoming-outgoing)/abs(incoming)})
            previous = temperature
    folder = cfg.study_dir/'mesh_convergence'
    folder.mkdir(parents=True, exist_ok=True)
    # CSV writer is local to avoid importing torch in an FE process.
    import csv
    with (folder/'mesh_convergence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder/'mesh_convergence.json').write_text(json.dumps(rows, indent=2))
    print(f'Mesh convergence and heat balance: {folder}')
    return rows
