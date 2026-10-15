import sys
import unittest
from unittest.mock import patch
from run import arguments, settings_from_arguments
from src.config import Settings
from src.data.compose import get_composition_id


class CommandTests(unittest.TestCase):
    def test_study_name_and_rename_preview_arguments(self):
        with patch.object(sys, 'argv', ['run.py', 'rename-outputs', '--study-name', 'Paper_Study', '--dry-run']):
            args = arguments()
            cfg = settings_from_arguments(args)
        self.assertEqual(cfg.study_name, 'paper_study')
        self.assertTrue(args.dry_run)
        self.assertEqual(get_composition_id(cfg), get_composition_id(Settings()))

    def test_grid_report_uses_tuning_budget_and_parses_separate_phases(self):
        with patch.object(sys, 'argv', ['run.py', 'report-tuning', '--report-phase', 'figures',
                                      '--report-seed', '123', '--tuning-size', '100']):
            args = arguments()
            cfg = settings_from_arguments(args)
        self.assertEqual(cfg.train_size, 100)
        self.assertEqual(cfg.epochs, cfg.tuning_epochs)
        self.assertEqual(args.report_phase, 'figures')
        self.assertEqual(args.report_seed, 123)

    def test_preview_uses_configured_source_count(self):
        for stage in ('preview', 'export-ansys'):
            with patch.object(sys, 'argv', ['run.py', stage, '--family', 'circle']):
                cfg = settings_from_arguments(arguments())
            self.assertEqual(cfg.preview_dataset_size, cfg.geometry_cases_per_family['circle'])
            self.assertEqual(cfg.ansys_export_dataset_size, cfg.geometry_cases_per_family['circle'])

    def test_selecting_evaluation_family_keeps_model_identity(self):
        with patch.object(sys, "argv", ["run.py", "evaluate", "--family", "circle", "--case", "0"]):
            cfg = settings_from_arguments(arguments())
        self.assertEqual(get_composition_id(cfg), get_composition_id(Settings()))
        self.assertEqual(cfg.composed_evaluation_families, ("circle",))
        self.assertEqual(cfg.composed_test_case_numbers, {"circle": (0,)})

    def test_ood_selection_keeps_model_identity(self):
        with patch.object(sys, "argv", ["run.py", "evaluate-ood", "--ood-families", "star", "--ood-cases", "7"]):
            cfg = settings_from_arguments(arguments())
        self.assertEqual(get_composition_id(cfg), get_composition_id(Settings()))
        self.assertEqual(cfg.ood_evaluation_families, ("star",))


if __name__ == "__main__":
    unittest.main()
