from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from src.config import Settings
from src.data.compose import compose_training_geometry_dataset, get_composition_id
from src.data.design import DESIGN, resolve_reference, geometry_key
from src.fem.geometry_spec import GeometryCase
from src.data.protocol import load_and_prepare_training_dataset


def source_dataset(cfg, family="circle"):
    path = Path(cfg.geometry_dataset_path(family, 5))
    path.parent.mkdir(parents=True, exist_ok=True)
    roles = np.array(["val", "test"] + ["train"] * 5)
    coords = np.array([[0, 0], [cfg.Lx, 0], [0, cfg.Ly], [cfg.Lx, cfg.Ly]], dtype=np.float32)
    sdf = np.tile([1, -1, 1, 1], (7, 1)).astype(np.float32)
    temperatures = np.tile([310, 0, 320, 330], (7, 1)).astype(np.float32)
    temperatures[0, 0] = 1000  # holdout extreme must never affect normalization
    metadata = [json.dumps(GeometryCase(family, .01 + i*.001, .01, parameters={"diameter": .008}).to_record()) for i in range(7)]
    records = dict(schema_version=[2], design=[DESIGN], geometry_family=[family],
                   geometry_group=["training"], training_eligible=[True], n_cases=[7],
                   sdf=sdf, mask=(sdf >= 0).astype(np.float32), temp_physical=temperatures,
                   metadata_json=metadata, fe_paths=["fe_cases/case.npz"] * 7,
                   roles=roles, geometry_keys=[geometry_key(record) for record in metadata],
                   candidate_ids=np.arange(7), case_ids=np.arange(7), coords=coords)
    for name in ("Lx", "Ly", "Nx", "Ny", "maxh", "order", "kappa", "h_hole", "T_inf", "q_top"):
        records[name] = [getattr(cfg, name)]
    np.savez(path, **records)
    (path.parent / "fe_cases").mkdir()
    np.savez(path.parent / "fe_cases/case.npz", test=[1])
    return path


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = Settings(data_root=str(self.root / "data"), results_root=str(self.root / "outputs"), Nx=2, Ny=2,
                            training_geometry_counts={"circle": 2}, training_geometry_source_sizes={"circle": 5},
                            validation_cases_per_family={"circle": 1}, test_cases_per_family={"circle": 1})
        source_dataset(self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_composition_and_training_only_scale(self):
        first = compose_training_geometry_dataset(self.cfg)
        prepared = load_and_prepare_training_dataset(self.cfg)
        self.assertEqual(prepared["temperature_min"], 310)
        self.assertEqual(prepared["temperature_max"], 330)
        self.assertGreater(prepared["target_normalized"][prepared["val_idx"]].max(), 1)
        with np.load(first, allow_pickle=True) as data:
            small = {role: data["geometry_keys"][data["roles"] == role].tolist() for role in ("train", "val", "test")}
        cfg = replace(self.cfg, training_geometry_counts={"circle": 4})
        second = compose_training_geometry_dataset(cfg)
        with np.load(second, allow_pickle=True) as data:
            for role in ("val", "test"):
                self.assertEqual(small[role], data["geometry_keys"][data["roles"] == role].tolist())
            self.assertEqual(small["train"], data["geometry_keys"][data["roles"] == "train"][:2].tolist())

    def test_portable_identity_and_paths(self):
        original = compose_training_geometry_dataset(self.cfg)
        relocated = replace(self.cfg, data_root=str(self.root / "moved/data"), results_root=str(self.root / "moved/outputs"))
        self.assertEqual(get_composition_id(self.cfg), get_composition_id(relocated))
        shutil.copytree(self.root / "data", self.root / "moved/data")
        path = relocated.composed_dataset_path(get_composition_id(relocated))
        with np.load(path, allow_pickle=True) as data:
            for reference in data["fe_paths"]:
                self.assertTrue(Path(resolve_reference(path, reference)).is_file())
        load_and_prepare_training_dataset(relocated)

    def test_ood_and_old_protocol_rejected(self):
        path = Path(self.cfg.geometry_dataset_path("circle", 5))
        with np.load(path) as saved:
            records = dict(saved)
        records["training_eligible"] = [False]
        np.savez(path, **records)
        with self.assertRaisesRegex(ValueError, "OOD"):
            compose_training_geometry_dataset(self.cfg)
        records["training_eligible"] = [True]
        records["schema_version"] = [1]
        np.savez(path, **records)
        with self.assertRaisesRegex(ValueError, "Regenerate"):
            compose_training_geometry_dataset(self.cfg)

    def test_tampered_split_rejected(self):
        path = compose_training_geometry_dataset(self.cfg)
        with np.load(path, allow_pickle=True) as saved:
            records = dict(saved)
        records["train_idx"] = records["test_idx"]
        np.savez(path, **records)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_and_prepare_training_dataset(self.cfg)


if __name__ == "__main__":
    unittest.main()
