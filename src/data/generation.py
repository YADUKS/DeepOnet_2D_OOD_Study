"""Shared FE generation/storage for role-labelled simple and OOD datasets."""
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.data.design import DESIGN, OOD_DESIGN, generation_settings, relative_path, geometry_key, assert_unique_geometries
from src.fem.fe_storage import save_geometry_fe_case
from src.fem.sampling import make_cartesian_grid


def generate_family(cfg, family, path, quotas, planner, solve, *, ood=False):
    path = Path(path)
    settings = generation_settings(cfg, family, ood)
    generation = {"settings": settings, "quotas": quotas}
    identity = json.dumps(generation, sort_keys=True)
    if path.exists() and not cfg.force_generation:
        with np.load(path, allow_pickle=True) as data:
            if "generation_identity" not in data or str(data["generation_identity"][0]) != identity:
                raise ValueError(f"Existing dataset has different settings: {path}")
            assert_unique_geometries(data["geometry_keys"], data["roles"], data["metadata_json"])
            previous = getattr(cfg, "_generation_seen", set())
            if previous.intersection(data["geometry_keys"].tolist()):
                raise ValueError("A reused family dataset duplicates another selected family. Review geometry definitions.")
            previous.update(data["geometry_keys"].tolist())
            for reference in data["fe_paths"]:
                if not (path.parent / str(reference)).is_file():
                    raise FileNotFoundError(f"Missing FE case in {path}; regenerate with --force.")
        print(f"Reusing completed dataset: {path}")
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: [] for key in ("sdf", "mask", "temp_physical", "metadata_json", "fe_paths",
                                  "candidate_ids", "rotation", "centres", "area", "perimeter",
                                  "domain_clearance", "roles", "role_case_ids", "geometry_keys")}
    failures = []
    seen = getattr(cfg, "_generation_seen", set())
    for role, quota in quotas.items():
        if quota < 1:
            raise ValueError(f"{family}/{role} count must be positive.")
        accepted = 0
        for planned in planner(cfg, family, quota, role):
            if accepted == quota:
                break
            case = replace(planned, case_id=len(arrays["sdf"]))
            # Ignore bookkeeping when detecting identical geometries across roles.
            try:
                key = geometry_key(case.to_record())
                if key in seen:
                    raise ValueError("Duplicate geometry across dataset roles.")
                # Cache by geometry under the size-independent design folder.
                # Appending a larger Halton pool/restarting does not repeat FE solves.
                cache_dir = path.parent.parent / "case_cache" / key
                cache = cache_dir / "grid.npz"
                if cache.exists() and not cfg.force_generation:
                    with np.load(cache, allow_pickle=False) as saved:
                        sdf, mask, temperature = (saved[name] for name in ("sdf", "mask", "temperature"))
                        diagnostics = json.loads(str(saved["diagnostics"][0]))
                        fe_path = str(cache_dir / str(saved["fe_file"][0]))
                    with np.load(fe_path, allow_pickle=True) as saved:
                        temperature_fe = saved["temperature"]
                else:
                    (_, _, sdf, mask, temperature, points, triangles, temperature_fe,
                     boundary_points, diagnostics) = solve(cfg, case)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    fe_path = save_geometry_fe_case(str(cache_dir), case, points, triangles,
                                                   temperature_fe, boundary_points, diagnostics)
                solid = temperature[mask > 0.5]
                if not len(solid) or not np.isfinite(solid).all() or not np.isfinite(temperature_fe).all():
                    raise ValueError("Nonfinite FE temperature at a solid sample or mesh vertex.")
                if not np.any(mask == 0):
                    raise ValueError("Grid does not resolve the hollow region; increase Nx/Ny.")
                if not cache.exists() or cfg.force_generation:
                    temporary = cache.with_suffix(".tmp.npz")
                    np.savez(temporary, sdf=sdf, mask=mask, temperature=temperature,
                             diagnostics=[json.dumps(diagnostics)], fe_file=[Path(fe_path).name])
                    temporary.replace(cache)
            except Exception as error:
                failures.append({"role": role, "candidate_id": case.candidate_id,
                                 "error": f"{type(error).__name__}: {error}"})
                continue
            seen.add(key)
            values = {
                "sdf": (sdf / max(cfg.Lx, cfg.Ly)).ravel(), "mask": mask.ravel(),
                "temp_physical": np.where(mask > 0.5, temperature, 0.0).ravel(),
                "metadata_json": json.dumps(case.to_record(), sort_keys=True),
                "fe_paths": relative_path(fe_path, path), "candidate_ids": case.candidate_id,
                "rotation": case.rotation, "centres": (case.cx, case.cy),
                "roles": role, "role_case_ids": accepted, "geometry_keys": key,
                **{name: diagnostics[name] for name in ("area", "perimeter", "domain_clearance")},
            }
            for name, value in values.items():
                arrays[name].append(value)
            accepted += 1
            print(f"[{family}/{role} {accepted}/{quota}] candidate={case.candidate_id}", flush=True)
        if accepted != quota:
            (path.parent / "generation_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            raise RuntimeError(f"Only {accepted}/{quota} valid {family}/{role} cases. Inspect generation_failures.json.")
    strings = {"metadata_json", "fe_paths", "roles", "geometry_keys"}
    integers = {"candidate_ids", "role_case_ids"}
    packed = {name: np.asarray(values, dtype=str if name in strings else np.int64 if name in integers else np.float32)
              for name, values in arrays.items()}
    valid = packed["temp_physical"][packed["mask"] > 0.5]
    lo, hi = float(valid.min()), float(valid.max())
    total = len(packed["sdf"])
    packed.update(schema_version=np.array([2]), design=np.array([OOD_DESIGN if ood else DESIGN]),
                  generation_identity=np.array([identity]), geometry_family=np.array([family]),
                  geometry_group=np.array(["ood" if ood else "training"]),
                  training_eligible=np.array([not ood]), case_ids=np.arange(total), n_cases=np.array([total]),
                  coords=make_cartesian_grid(cfg)[-1], T_min=np.array([lo]), T_max=np.array([hi]),
                  temp=((packed["temp_physical"] - lo) / (hi - lo + 1e-12) * packed["mask"]).astype(np.float32),
                  normalization_scope=np.array(["compatibility_only_do_not_use_for_training_or_evaluation"]),
                  sdf_normalization_length=np.array([max(cfg.Lx, cfg.Ly)]))
    for name in ("Lx", "Ly", "Nx", "Ny", "maxh", "order", "kappa", "h_hole", "T_inf", "q_top"):
        packed[name] = np.array([getattr(cfg, name)])
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **packed)
    temporary.replace(path)
    manifest = {"schema_version": 2, **generation, "total_cases": total,
                "overlap_check": assert_unique_geometries(packed["geometry_keys"], packed["roles"], packed["metadata_json"]),
                "accepted_counts": quotas, "rejections": failures, "config": asdict(cfg),
                "sampling": {role: "halton_unscrambled_start_1" if role == "train" else "lhs_primary_plus_reserve"
                             for role in quotas}}
    (path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved dataset: {path}")
    return str(path)
