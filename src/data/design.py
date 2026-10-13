"""Parameter-space designs: nested Halton training, independent LHS evaluation.

Halton is unscrambled and begins at index 1, as in the reference study. LHS
uses a seeded generator. FE grid sampling remains in src/fem/sampling.py.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

DESIGN = "halton_train_lhs_holdouts_v2"
OOD_DESIGN = "lhs_ood_v2"
ROLES = ("train", "val", "test")
ROLE_SEED_OFFSETS = {"train": 0, "val": 1000000, "test": 2000000, "ood": 3000000}


def halton_sequence(n_points, n_dim, start=1):
    """Return a size-independent prefix using successive prime bases."""
    if n_points < 0 or n_dim < 1 or start < 1:
        raise ValueError("Halton requires n_points >= 0, n_dim >= 1, start >= 1.")
    primes = []
    candidate = 2
    while len(primes) < n_dim:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    result = np.zeros((n_points, n_dim), dtype=np.float64)
    for column, base in enumerate(primes):
        indices = np.arange(start, start + n_points, dtype=np.int64)
        factor = 1.0 / base
        while np.any(indices):
            result[:, column] += (indices % base) * factor
            indices //= base
            factor /= base
    return result


def latin_hypercube_sampling(n_samples, n_dim, seed=42):
    if n_samples < 0 or n_dim < 1:
        raise ValueError("LHS requires n_samples >= 0 and n_dim >= 1.")
    rng = np.random.default_rng(int(seed))
    samples = np.empty((n_samples, n_dim), dtype=np.float64)
    for dimension in range(n_dim):
        values = (np.arange(n_samples) + rng.random(n_samples)) / max(n_samples, 1)
        rng.shuffle(values)
        samples[:, dimension] = values
    return samples


def parameter_candidates(n_cases, n_dim, role, seed, multiplier=2.0):
    if n_cases < 1 or role not in (*ROLES, "ood"):
        raise ValueError("A positive case count and a known dataset role are required.")
    reserve = max(30, int(np.ceil(max(multiplier - 1.0, 0.0) * n_cases)))
    if role == "train":
        return halton_sequence(n_cases + reserve, n_dim)
    # Preserve an exact quota-sized LHS; reserves replace rejected candidates only.
    return np.vstack((latin_hypercube_sampling(n_cases, n_dim, seed),
                      latin_hypercube_sampling(reserve, n_dim, seed + 5000000)))


def generation_settings(cfg, family, ood=False):
    """Only settings affecting geometry/FE data; excludes paths and run choices."""
    fixed = {"Lx", "Ly", "Nx", "Ny", "maxh", "order", "kappa", "h_hole",
             "T_inf", "q_top", "random_position", "cx_fixed", "cy_fixed",
             "curved_boundary_segments", "seed"}
    if family == "circle":
        fixed.update(("D_min", "D_max"))
    elif family == "rectangle":
        fixed.update(("W_min", "W_max", "H_min", "H_max"))
    family_prefix = "fin_" if family == "fin_like" else family + "_"
    prefixes = (family_prefix, "geometry_rotation_", "geometry_minimum_")
    settings = {key: value for key, value in asdict(cfg).items()
                if key in fixed or key.startswith(prefixes)}
    from src.fem.registry import custom_shape
    spec = custom_shape(family)
    if spec:
        import inspect
        settings["custom_geometry"] = {"version": spec["version"], "dimensions": spec["dimensions"],
            "ranges": cfg.custom_geometry_ranges.get(family, {}),
            "mapper": inspect.getsource(spec["map_parameters"]), "boundary": inspect.getsource(spec["boundary"])}
    settings.update(geometry_identity="canonical_boundary_v1", design=OOD_DESIGN if ood else DESIGN, family=family,
                    family_seed_offset=int(cfg.geometry_seed_offsets[family]),
                    candidate_multiplier=(cfg.ood_candidate_multiplier if ood
                                          else cfg.geometry_candidate_multiplier))
    if not ood:
        settings.update(val_cases=int(cfg.validation_cases_per_family[family]),
                        test_cases=int(cfg.test_cases_per_family[family]))
    return settings


def design_id(cfg, family, ood=False):
    raw = json.dumps(generation_settings(cfg, family, ood), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def relative_path(path, owner):
    """Portable POSIX reference relative to the file containing the reference."""
    import os
    try:
        return Path(os.path.relpath(path, Path(owner).parent)).as_posix()
    except ValueError:
        # Windows cannot express a relative reference between different drives.
        return Path(path).as_posix()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_reference(owner, reference):
    # Old Windows separators are accepted for relative legacy references too.
    path = Path(str(reference).replace("\\", "/"))
    return str(path if path.is_absolute() else (Path(owner).parent / path).resolve())


def role_splits(roles, families):
    """Validate frozen membership and require every family in every role."""
    roles, families = np.asarray(roles), np.asarray(families)
    if roles.ndim != 1 or roles.shape != families.shape or not len(roles):
        raise ValueError("Roles and families must have one label per case.")
    if not np.isin(roles, ROLES).all():
        raise ValueError("Training data has unknown or OOD dataset roles.")
    counts = {}
    for family in sorted(set(families.tolist())):
        counts[family] = {name: int(np.count_nonzero((families == family) & (roles == role)))
                          for name, role in (("train", "train"), ("validation", "val"), ("test", "test"))}
        if min(counts[family].values()) < 1:
            raise ValueError(f"Family {family} requires nonempty train, val and test sets.")
        counts[family]["total"] = int(np.count_nonzero(families == family))
    return *(np.flatnonzero(roles == role) for role in ROLES), counts


def geometry_key(record):
    """Canonical geometry identity, with coordinates quantized at 1e-11 m.

    Removes bookkeeping, circle/ellipse angular symmetries and polygon vertex
    ordering. Distinct nearby geometries remain valid study samples.
    """
    from src.fem.geometry_spec import GeometryCase, transform_points
    if isinstance(record, str):
        record = json.loads(record)
    case = GeometryCase(**record)
    q = lambda a: np.rint(np.asarray(a, dtype=float) / 1e-11).astype(np.int64).tolist()
    if case.family == "circle":
        payload = ["circle", q([case.cx, case.cy, case.scale * case.parameters["diameter"] / 2])]
    elif case.family == "ellipse":
        major, minor = (case.scale * case.parameters[k] / 2 for k in ("major_axis", "minor_axis"))
        if q(major) == q(minor):
            payload = ["circle", q([case.cx, case.cy, major])]
        else:
            c, s = np.cos(case.rotation), np.sin(case.rotation)
            matrix = [major*c*c + minor*s*s, (major-minor)*s*c, major*s*s+minor*c*c]
            payload = ["ellipse", q([case.cx, case.cy, *matrix])]
    else:
        if case.group == "ood":
            from src.fem.ood_geometries import build_canonical_ood_boundary as boundary
        else:
            from src.fem.simple_geometries import build_canonical_simple_boundary as boundary
        points = transform_points(boundary(case, curve_segments=96), case.cx, case.cy, case.rotation, case.scale)
        a, b = points-np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)-points
        keep = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]) > 1e-18
        payload = ["boundary", sorted(map(tuple, q(points[keep])))]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_unique_geometries(keys, roles, metadata=None):
    keys, roles = np.asarray(keys), np.asarray(roles)
    if keys.ndim != 1 or keys.shape != roles.shape:
        raise ValueError("Geometry identities and roles must have one entry per case.")
    if metadata is not None:
        if len(metadata) != len(keys):
            raise ValueError("Geometry metadata count mismatch.")
        reconstructed = np.array([geometry_key(str(record)) for record in metadata])
        if not np.array_equal(keys, reconstructed):
            raise ValueError("Geometry identities disagree with saved metadata. Regenerate this dataset.")
    seen = {}
    for index, (key, role) in enumerate(zip(keys, roles)):
        if str(key) in seen:
            other, other_role = seen[str(key)]
            raise ValueError(f"Duplicate geometry: cases {other} ({other_role}) and {index} ({role}).")
        seen[str(key)] = (index, str(role))
    return {"checked_cases": len(keys), "duplicate_geometries": 0,
            "identity": "canonical_boundary_v1", "coordinate_resolution_m": 1e-11}
