"""Frozen selected model, nested training sizes PER FAMILY, independent seeds."""
from pathlib import Path
import json
from .tuning import final_settings, run_cell
from src.data.design import file_digest

def train_scaling_run(cfg, size, seed):
    from src.analysis.metrics import evaluate
    run = final_settings(cfg, seed, size, scaling=True)
    run_cell(run)
    path = Path(run.outdir) / 'all_test_cases_grid_summary.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['checkpoint_sha256'] == file_digest(run.model_path):
            print(f'Skipping completed test evaluation: {run.run_name}')
            return
    evaluate(run, split='test')
