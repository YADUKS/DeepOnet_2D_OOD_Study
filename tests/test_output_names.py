"""Readable labels preserve data identity, old paths and existing file contents."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import Settings
from src.data.compose import get_composition_id
from src.data.design import design_id, file_digest
from src.analysis.output_names import CATEGORIES, rename_outputs, stage_name
from src.analysis.tuning_report import report_directory, validate_cache


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

    def snapshots(self):
        return {str(path.relative_to(self.root)): file_digest(path)
                for path in self.root.rglob('*') if path.is_file()}

    def test_merge_preview_then_move_preserves_legacy_new_and_dataset_files(self):
        files = self.legacy_files()
        current = self.cfg.output_root/'study_results'/self.cfg.case_tag
        # Merge at both root and run levels, as happens after a new report.
        same_run = current/self.cfg.run_name
        same_run.mkdir(parents=True)
        keep = same_run/'new_evaluation.csv'
        keep.write_bytes(b'new completed evaluation')
        data = Path(self.cfg.data_root)/'cached_source.npz'
        data.parent.mkdir()
        data.write_bytes(b'original dataset')
        before = self.snapshots()
        self.assertEqual(len(rename_outputs(self.cfg, merge_existing=True, dry_run=True)), 5)
        self.assertEqual(self.snapshots(), before)
        rename_outputs(self.cfg, merge_existing=True)
        for path in files:
            target = Path(str(path).replace(self.cfg.legacy_case_tag, self.cfg.case_tag))
            self.assertEqual(file_digest(target), before[str(path.relative_to(self.root))])
            self.assertFalse(path.exists())
        self.assertEqual(keep.read_bytes(), b'new completed evaluation')
        self.assertEqual(data.read_bytes(), b'original dataset')
        self.assertFalse((self.cfg.output_root/'study_results'/self.cfg.legacy_case_tag).exists())
        self.assertEqual(rename_outputs(self.cfg, merge_existing=True), [])

    def test_merge_identical_files_retains_both_copies_and_moves_missing_files(self):
        files = self.legacy_files()
        source = files[2]
        target = Path(str(source).replace(self.cfg.legacy_case_tag, self.cfg.case_tag))
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        extra = source.parent/'missing.csv'
        extra.write_bytes(b'old results missing in new folder')
        before = self.snapshots()
        rename_outputs(self.cfg, merge_existing=True, dry_run=True)
        self.assertEqual(self.snapshots(), before)
        rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(source.read_bytes(), target.read_bytes())
        self.assertEqual((target.parent/extra.name).read_bytes(), b'old results missing in new folder')
        self.assertFalse(extra.exists())
        after = self.snapshots()
        self.assertEqual(rename_outputs(self.cfg, merge_existing=True), [])
        self.assertEqual(self.snapshots(), after)

    def test_merge_different_files_stop_all_categories_even_in_preview(self):
        files = self.legacy_files()
        source = files[-1]
        target = Path(str(source).replace(self.cfg.legacy_case_tag, self.cfg.case_tag))
        target.parent.mkdir(parents=True)
        # Same length: comparison must use contents rather than only file size.
        target.write_bytes(b'x'*source.stat().st_size)
        before = self.snapshots()
        for preview in (True, False):
            with self.subTest(dry_run=preview), self.assertRaisesRegex(FileExistsError, 'Different contents'):
                rename_outputs(self.cfg, merge_existing=True, dry_run=preview)
            self.assertEqual(self.snapshots(), before)
            self.assertTrue(all(path.exists() for path in files))

    def test_merge_file_directory_conflict_leaves_all_outputs_untouched(self):
        files = self.legacy_files()
        source = files[-1]
        target = Path(str(source).replace(self.cfg.legacy_case_tag, self.cfg.case_tag))
        target.mkdir(parents=True)
        before = self.snapshots()
        with self.assertRaisesRegex(FileExistsError, 'File/directory mismatch'):
            rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(self.snapshots(), before)
        self.assertTrue(target.is_dir())
        self.assertTrue(all(path.exists() for path in files))

    def test_merge_nested_source_and_destination_links_rejected(self):
        files = self.legacy_files()
        source = files[-1]
        target = Path(str(source).replace(self.cfg.legacy_case_tag, self.cfg.case_tag))
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        original = Path.is_symlink
        before = self.snapshots()
        for linked in (source, target):
            def is_symlink(path):
                return path == linked or original(path)
            with self.subTest(linked=linked), patch.object(Path, 'is_symlink', is_symlink):
                with self.assertRaisesRegex(ValueError, 'linked output'):
                    rename_outputs(self.cfg, merge_existing=True)
            self.assertEqual(self.snapshots(), before)

    def test_merge_nested_windows_junction_rejected(self):
        files = self.legacy_files()
        linked = files[-1].parent
        before = self.snapshots()
        with patch.object(Path, 'is_junction', lambda path: path == linked, create=True):
            with self.assertRaisesRegex(ValueError, 'linked output'):
                rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(self.snapshots(), before)

    def test_merge_failure_rolls_back_earlier_moves(self):
        files = self.legacy_files()
        current = self.cfg.output_root/'study_results'/self.cfg.case_tag
        current.mkdir(parents=True)
        before = self.snapshots()
        original = Path.rename
        calls = []
        def failing_rename(path, destination):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('simulated filesystem failure')
            return original(path, destination)
        with patch.object(Path, 'rename', failing_rename):
            with self.assertRaisesRegex(OSError, 'simulated filesystem failure'):
                rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(self.snapshots(), before)
        self.assertTrue(all(path.exists() for path in files))

    def test_merge_concurrent_destination_is_not_overwritten(self):
        files = self.legacy_files()
        before = self.snapshots()
        concurrent = self.cfg.output_root/'logs'/self.cfg.case_tag
        original = Path.rename
        calls = []
        def concurrent_rename(path, destination):
            result = original(path, destination)
            calls.append(path)
            if len(calls) == 1:
                concurrent.mkdir(parents=True)
                (concurrent/'keep.bin').write_bytes(b'concurrent writer')
            return result
        with patch.object(Path, 'rename', concurrent_rename):
            with self.assertRaisesRegex(FileExistsError, 'destination appeared'):
                rename_outputs(self.cfg, merge_existing=True)
        self.assertTrue(all(path.exists() for path in files))
        self.assertEqual((concurrent/'keep.bin').read_bytes(), b'concurrent writer')
        for relative, digest in before.items():
            self.assertEqual(file_digest(self.root/relative), digest)

    def test_merge_late_path_check_failure_rolls_back(self):
        from src.analysis.output_names import _check_child
        files = self.legacy_files()
        before = self.snapshots()
        moved_root = self.cfg.output_root/'checkpoints'/self.cfg.case_tag
        rejected = self.cfg.output_root/'logs'/self.cfg.legacy_case_tag
        def reject_after_move(path, parent, root):
            if path == rejected and moved_root.exists():
                raise ValueError('simulated changed path')
            return _check_child(path, parent, root)
        with patch('src.analysis.output_names._check_child', reject_after_move):
            with self.assertRaisesRegex(ValueError, 'simulated changed path'):
                rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(self.snapshots(), before)
        self.assertTrue(all(path.exists() for path in files))

    def test_merge_report_cache_remains_discoverable_and_passes_hash_checks(self):
        suffix = Path('tanh_grid/N0200_per_family/seeds_123/model_seed123')
        source = self.cfg.output_root/'reports'/self.cfg.legacy_case_tag/suffix
        target = self.cfg.output_root/'reports'/self.cfg.case_tag/suffix
        payload = source/'data/predictions.npz'
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b'cached prediction bytes')
        manifest = {'schema': 1, 'evaluation_complete': True,
                    'cached_files': {'data/predictions.npz': file_digest(payload)}}
        (source/'report_manifest.json').write_text(json.dumps(manifest))
        target.mkdir(parents=True)
        (target/'new_figure.pdf').write_bytes(b'existing readable result')
        digest = file_digest(source/'report_manifest.json')
        rename_outputs(self.cfg, merge_existing=True)
        self.assertEqual(report_directory(self.cfg, 'grid', [123], 123), target)
        validate_cache(target, json.loads((target/'report_manifest.json').read_text()))
        self.assertEqual(file_digest(target/'report_manifest.json'), digest)
        self.assertEqual((target/'new_figure.pdf').read_bytes(), b'existing readable result')

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
