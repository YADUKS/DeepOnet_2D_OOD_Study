"""Readable labels preserve data identity, old paths and existing file contents."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.config import Settings
from src.data.compose import get_composition_id
from src.data.design import design_id, file_digest
from src.analysis.output_names import CATEGORIES, rename_outputs, stage_name
from src.analysis.tuning_report import report_directory


class OutputNamesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = Settings(data_root=str(self.root/'data'), results_root=str(self.root/'outputs'),
                            run_name='tune_deeponet_N0200_grid_L2_N10_seed123')

    def tearDown(self):
        self.temp.cleanup()

    def legacy_files(self):
        files = []
        for category in CATEGORIES:
            path = self.cfg.output_root/category/self.cfg.legacy_case_tag/self.cfg.run_name/'unchanged.bin'
            path.parent.mkdir(parents=True)
            path.write_bytes((category+' scientific data').encode())
            files.append(path)
        return files

    def test_label_describes_study_families_pool_and_unique_identity(self):
        self.assertEqual(self.cfg.case_tag, 'deeponet_2d_ood_9families_pool800_'+get_composition_id(self.cfg))
        custom = replace(self.cfg, study_name='My_Heat_Study', training_geometry_counts={'circle': 50, 'ellipse': 100})
        self.assertTrue(custom.case_tag.startswith('my_heat_study_2families_pooltotal150_'))
        changed_physics = replace(self.cfg, kappa=self.cfg.kappa+1)
        self.assertNotEqual(self.cfg.case_tag, changed_physics.case_tag)

    def test_name_is_independent_of_dataset_and_generation_identities(self):
        renamed = replace(self.cfg, study_name='paper_experiment')
        self.assertNotEqual(self.cfg.case_tag, renamed.case_tag)
        self.assertEqual(get_composition_id(self.cfg), get_composition_id(renamed))
        for family, ood in [('circle', False), ('star', True)]:
            self.assertEqual(design_id(self.cfg, family, ood), design_id(renamed, family, ood))
        self.assertEqual(self.cfg.dataset_path, renamed.dataset_path)
        self.assertEqual(self.cfg.geometry_dataset_path('circle', 800), renamed.geometry_dataset_path('circle', 800))
        self.assertEqual(self.cfg.ood_dataset_path('star', 20), renamed.ood_dataset_path('star', 20))

    def test_bad_or_unsafe_names_rejected(self):
        for name in ('../other', 'other/study', 'other\\study', 'with space', '.', 'x'*49, 7):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'study_name'):
                replace(self.cfg, study_name=name)
        self.assertEqual(replace(self.cfg, study_name='').case_tag, self.cfg.legacy_case_tag)

    def test_legacy_run_directories_are_found_without_moving_files(self):
        self.legacy_files()
        for path in (self.cfg.checkpoint_outdir, self.cfg.log_outdir, self.cfg.figure_outdir, self.cfg.outdir):
            self.assertIn(self.cfg.legacy_case_tag, str(path))
            self.assertTrue((Path(path)/'unchanged.bin').is_file())
        new_run = replace(self.cfg, run_name='new_grid_cell')
        self.assertIn(self.cfg.case_tag, new_run.checkpoint_outdir)
        self.assertIn(self.cfg.case_tag, new_run.outdir)

    def test_new_paths_take_precedence_without_merging_roots(self):
        self.legacy_files()
        current = self.cfg.output_root/'checkpoints'/self.cfg.case_tag/self.cfg.run_name
        current.mkdir(parents=True)
        self.assertEqual(Path(self.cfg.checkpoint_outdir), current)

    def test_preview_and_rename_preserve_every_file_and_datasets(self):
        files = self.legacy_files()
        before = {str(p.relative_to(self.cfg.output_root)): file_digest(p) for p in files}
        data = Path(self.cfg.data_root)/'cached_source.npz'
        data.parent.mkdir()
        data.write_bytes(b'original dataset bytes')
        original_data = file_digest(data)
        plan = rename_outputs(self.cfg, dry_run=True)
        self.assertEqual(len(plan), 5)
        self.assertTrue(all(p.exists() for p in files))
        rename_outputs(self.cfg)
        for relative, digest in before.items():
            new = self.cfg.output_root/relative.replace(self.cfg.legacy_case_tag, self.cfg.case_tag)
            self.assertEqual(file_digest(new), digest)
        self.assertEqual(file_digest(data), original_data)
        self.assertEqual(rename_outputs(self.cfg), [])
        self.assertIn(self.cfg.case_tag, self.cfg.checkpoint_outdir)

    def test_destination_conflict_stops_all_moves(self):
        files = self.legacy_files()
        destination = self.cfg.output_root/'reports'/self.cfg.case_tag
        destination.mkdir(parents=True)
        keep = destination/'keep.bin'
        keep.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            rename_outputs(self.cfg)
        self.assertTrue(all(p.exists() for p in files))
        self.assertEqual(keep.read_bytes(), b'keep')

    def test_symbolic_link_escape_rejected(self):
        # Mock only resolution so the containment check runs on Windows without
        # requiring privileges for actual symlinks/junctions.
        from unittest.mock import patch
        original = Path.resolve
        escaped = self.root/'outside'
        checkpoint_root = self.cfg.output_root/'checkpoints'
        def resolve(path, *args, **kwargs):
            return escaped if path == checkpoint_root else original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), self.assertRaisesRegex(ValueError, 'intended directory'):
            rename_outputs(self.cfg)

    def test_report_paths_describe_stage_and_reuse_old_caches(self):
        current = report_directory(self.cfg, 'grid', [123], 123)
        self.assertIn('tanh_neurons_layers_grid', current.parts)
        self.assertIn('train200_per_family', current.parts)
        self.assertEqual(stage_name('lr'), 'learning_rate_study')
        old = (self.cfg.output_root/'reports'/self.cfg.legacy_case_tag/'tanh_grid' /
               'N0200_per_family'/'seeds_123'/'model_seed123')
        old.mkdir(parents=True)
        (old/'report_manifest.json').write_text('{"evaluation_complete": true}')
        self.assertEqual(report_directory(self.cfg, 'grid', [123], 123), old)
        rename_outputs(self.cfg)
        moved = self.cfg.output_root/'reports'/self.cfg.case_tag/old.relative_to(self.cfg.output_root/'reports'/self.cfg.legacy_case_tag)
        self.assertEqual(report_directory(self.cfg, 'grid', [123], 123), moved)
        current.mkdir(parents=True)
        self.assertEqual(report_directory(self.cfg, 'grid', [123], 123), current)


if __name__ == '__main__':
    unittest.main()
