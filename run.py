"""Single entry point: python run.py STAGE [OPTIONS]. See --help and README.md."""
import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import subprocess
import sys

from src.config import PROJECT_ROOT, Settings
from src.fem.geometry_spec import TRAINING_GEOMETRY_FAMILIES, OOD_GEOMETRY_FAMILIES, canonical_geometry_name

STAGES = ("generate", "compose", "train", "evaluate", "generate-ood", "evaluate-ood",
          "preview", "export-ansys", "plot-sampling", "config", "gpu-info", "smoke", "mesh-check", "plot", "benchmark",
          "tune", "tune-select", "report-tuning", "train-final", "train-scaling", "summarize", "compare", "scaling-report",
          "rename-outputs")


def arguments():
    parser = argparse.ArgumentParser(description="DeepONet geometry OOD study: Halton training + fixed LHS holdouts")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, help="JSON Settings overrides; paths resolve from the repository")
    parser.add_argument("--families", help="Comma-separated simple families; also selects composition/evaluation families")
    parser.add_argument("--ood-families", help="Comma-separated OOD families")
    parser.add_argument("--cases", type=int, help="Halton training-pool size per selected simple family")
    parser.add_argument("--train-count", type=int, help="Training prefix to compose per selected family (default: source size)")
    parser.add_argument("--val-cases", type=int, help="Fixed LHS validation count per family")
    parser.add_argument("--test-cases", type=int, help="Fixed LHS held-out test count per family")
    parser.add_argument("--ood-cases", type=int, help="Independent LHS cases per selected OOD family")
    parser.add_argument("--composition-id", help="Use a previously composed dataset/model ID")
    parser.add_argument("--run-name", help="Separate model run inside the composition results")
    parser.add_argument("--study-name", help="Readable study folder prefix; data/checkpoint identity is unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Preview rename-outputs without moving folders")
    parser.add_argument("--data-root", help="Dataset root, absolute or relative to this repository")
    parser.add_argument("--results-root", help="Results root, absolute or relative to this repository")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, help="Model training seed")
    parser.add_argument("--data-seed", type=int, help="LHS seed (unscrambled Halton is seed-independent)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--train-size", type=int, help="Training cases PER FAMILY; does not change the saved master dataset")
    parser.add_argument("--tuning-size", type=int, help="Tuning cases per family (default 200)")
    parser.add_argument("--sizes", help="Comma-separated training sizes per family")
    parser.add_argument("--seeds", help="Comma-separated model seeds; tuning defaults to 123")
    parser.add_argument("--stage-name", choices=("grid", "grid_gelu", "lr"))
    parser.add_argument("--report-phase", choices=("all", "evaluate", "figures"), default="all",
                        help="Tuning report: cache predictions, render cached figures, or both")
    parser.add_argument("--report-seed", type=int, help="Saved winner checkpoint seed; defaults to the first tuning seed")
    parser.add_argument("--report-dir", type=Path, help="Optional separate tuning report directory")
    parser.add_argument("--cell", type=int, help="One tuning cell; defaults to SLURM_ARRAY_TASK_ID when present")
    parser.add_argument("--branch-hidden", help="Comma-separated branch hidden widths")
    parser.add_argument("--trunk-hidden", help="Comma-separated trunk hidden widths")
    parser.add_argument("--activation", choices=("tanh", "gelu"))
    parser.add_argument("--point-chunk", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--early-stop", type=int)
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--benchmark-epochs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--maxh", type=float)
    parser.add_argument("--force", action="store_true", help="Regenerate data; model runs require --resume or a new --run-name")
    parser.add_argument("--show", action="store_true", help="Show interactive figures locally")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--plot-cases", type=int, help="Number of plotted cases per family; zero scores without case figures")
    parser.add_argument("--case", type=int, help="Zero-based case to preview/export or held-out case to evaluate")
    parser.add_argument("--family", help="One family for preview/export or evaluation (does not alter composition)")
    parser.add_argument("--group", choices=("training", "ood", "composed"), default="training")
    parser.add_argument("--individual-figures", action="store_true")
    return parser.parse_args()


def selected_names(text, allowed):
    names = tuple(canonical_geometry_name(name) for name in text.split(","))
    if not names or len(set(names)) != len(names) or not set(names).issubset(allowed):
        raise ValueError(f"Choose distinct families from: {', '.join(allowed)}")
    return names


def settings_from_arguments(args):
    overrides = {}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(overrides) - {field.name for field in fields(Settings)}
        if unknown:
            raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")
    cfg = Settings(**overrides)
    simple = selected_names(args.families, TRAINING_GEOMETRY_FAMILIES) if args.families else tuple(cfg.generate_geometry_families)
    ood = selected_names(args.ood_families, OOD_GEOMETRY_FAMILIES) if args.ood_families else tuple(cfg.generate_ood_families)
    if args.families:
        cfg.generate_geometry_families = simple
        cfg.composed_evaluation_families = simple
        cfg.training_geometry_counts = {family: cfg.training_geometry_counts.get(family, 0) if family in simple else 0
                                        for family in TRAINING_GEOMETRY_FAMILIES}
    if args.ood_families:
        cfg.generate_ood_families = cfg.ood_evaluation_families = ood
    for family in simple:
        if args.cases is not None:
            cfg.geometry_cases_per_family[family] = args.cases
            cfg.training_geometry_source_sizes[family] = args.cases
            cfg.training_geometry_counts[family] = args.cases
        if args.train_count is not None:
            cfg.training_geometry_counts[family] = args.train_count
        if args.val_cases is not None:
            cfg.validation_cases_per_family[family] = args.val_cases
        if args.test_cases is not None:
            cfg.test_cases_per_family[family] = args.test_cases
    for family in ood:
        if args.ood_cases is not None:
            cfg.ood_cases_per_family[family] = cfg.ood_evaluation_source_sizes[family] = args.ood_cases
    mappings = {"composition_id": "training_composition_id", "seed": "training_seed", "data_seed": "seed",
                "nx": "Nx", "ny": "Ny", "dpi": "publish_dpi"}
    for name in ("composition_id", "seed", "data_seed", "nx", "ny", "maxh", "dpi", "data_root", "results_root",
                 "run_name", "study_name", "device", "cpu_threads", "epochs", "batch_size", "lr", "latent_dim"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, mappings.get(name, name), value)
    for argument, field in (("train_size", "train_size"), ("tuning_size", "tuning_size"),
                            ("point_chunk", "point_chunk"), ("checkpoint_every", "checkpoint_every"),
                            ("early_stop", "early_stop_patience"), ("scheduler_patience", "scheduler_patience"),
                            ("activation", "deeponet_activation")):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, value)
    for argument, field in (("sizes", "scaling_sizes"), ("branch_hidden", "branch_hidden"), ("trunk_hidden", "trunk_hidden")):
        if getattr(args, argument):
            value = tuple(int(x) for x in getattr(args, argument).split(','))
            if min(value) < 1:
                raise ValueError("Sizes and hidden widths must be positive.")
            setattr(cfg, field, value)
    cfg.resume = args.resume
    if args.stage in ("tune", "tune-select", "report-tuning"):
        if args.train_size is not None:
            cfg.tuning_size = args.train_size
        cfg.train_size = cfg.tuning_size
        cfg.epochs = args.epochs if args.epochs is not None else cfg.tuning_epochs
    selected_counts = [v for v in cfg.training_geometry_counts.values() if v > 0]
    if not selected_counts:
        raise ValueError("Select at least one training family.")
    if not cfg.run_name:
        cfg.run_name = f"deeponet_N{cfg.train_size or min(selected_counts):04d}_seed{cfg.training_seed}"
    for field in ("scaling_sizes", "study_seeds", "tuning_seeds", "tuning_layers", "tuning_neurons"):
        values = getattr(cfg, field)
        if not values or len(set(values)) != len(values) or any(int(v) != v or v < (0 if 'seeds' in field else 1) for v in values):
            raise ValueError(f"{field} must contain distinct valid integers.")
    if sorted(cfg.scaling_sizes) != list(cfg.scaling_sizes):
        raise ValueError("Training sizes must be in increasing order.")
    if cfg.train_size < 0 or min(cfg.tuning_size, cfg.point_chunk, cfg.checkpoint_every, cfg.early_stop_patience) < 1:
        raise ValueError("Training sizes/chunk/checkpoint/patience settings are invalid.")
    cfg.force_generation = args.force
    if args.show:
        cfg.show_figures = cfg.stored_preview_show = cfg.show_loss_plot = True
    if args.no_pdf:
        cfg.save_pdf_figures = cfg.ood_save_pdf_figures = cfg.composed_save_pdf_figures = cfg.stored_preview_save_pdf = False
    if args.plot_cases is not None:
        if args.plot_cases < 0:
            raise ValueError("--plot-cases cannot be negative.")
        cfg.composed_plot_cases_per_family = args.plot_cases
        cfg.ood_plot_case_numbers = tuple(range(args.plot_cases))
    cfg.stored_preview_save_individual = args.individual_figures
    if args.family and args.stage in ("evaluate", "plot"):
        cfg.composed_evaluation_families = selected_names(args.family, TRAINING_GEOMETRY_FAMILIES)
    elif args.family and args.stage == "evaluate-ood":
        cfg.ood_evaluation_families = selected_names(args.family, OOD_GEOMETRY_FAMILIES)
    if args.case is not None:
        if args.case < 0:
            raise ValueError("--case cannot be negative.")
        cfg.preview_dataset_case_number = cfg.ansys_export_case_number = args.case
        cfg.test_case_number = args.case
        if args.stage == "evaluate":
            cfg.test_case_number = args.case
            cfg.composed_test_case_numbers = {family: (args.case,) for family in cfg.composed_evaluation_families}
        elif args.stage == "evaluate-ood":
            cfg.ood_test_case_numbers = {family: (args.case,) for family in cfg.ood_evaluation_families}
    if args.stage in ("preview", "export-ansys"):
        if not args.family:
            raise ValueError("Stored preview/export requires --family.")
        if args.stage == "preview" and args.group == "composed":
            raise ValueError("Preview reads an independent source: use --group training or ood.")
        cfg.preview_dataset_family = cfg.ansys_export_family = canonical_geometry_name(args.family)
        cfg.preview_dataset_group = cfg.ansys_export_dataset_group = args.group
        size = args.ood_cases if args.group == "ood" else args.cases
        if size is None and args.group != 'composed':
            counts = cfg.ood_cases_per_family if args.group == 'ood' else cfg.geometry_cases_per_family
            size = counts[cfg.preview_dataset_family]
        if size is not None:
            cfg.preview_dataset_size = cfg.ansys_export_dataset_size = size
    for name in ("cpu_threads", "epochs", "batch_size", "latent_dim", "Nx", "Ny", "publish_dpi"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if min(cfg.Nx, cfg.Ny) < 2 or cfg.lr <= 0 or cfg.maxh <= 0:
        raise ValueError("Grid dimensions >= 2 and positive lr/maxh are required.")
    cfg.__post_init__()
    return cfg


def number_list(value):
    values = tuple(int(part.strip()) for part in value.split(","))
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError("Specify distinct nonnegative integers.")
    return values


def smoke():
    """Exercise every study stage using tiny data/grids; each runs in a fresh process."""
    import time
    workspace = PROJECT_ROOT / "verification" / ("study_smoke_" + str(time.time_ns()))
    workspace.mkdir(parents=True)
    config = {
        "data_root": str(workspace / "data"), "results_root": str(workspace / "outputs"),
        "generate_geometry_families": ["circle", "rectangle"],
        "training_geometry_counts": {"circle": 4, "rectangle": 4},
        "training_geometry_source_sizes": {"circle": 4, "rectangle": 4},
        "geometry_cases_per_family": {"circle": 4, "rectangle": 4},
        "validation_cases_per_family": {"circle": 2, "rectangle": 2},
        "test_cases_per_family": {"circle": 2, "rectangle": 2},
        "generate_ood_families": ["star"], "ood_evaluation_families": ["star"],
        "ood_cases_per_family": {"star": 2}, "ood_evaluation_source_sizes": {"star": 2},
        "Nx": 16, "Ny": 12, "maxh": .002, "device": "cpu", "cpu_threads": 2,
        "epochs": 2, "tuning_epochs": 2, "tuning_size": 2,
        "tuning_layers": [2], "tuning_neurons": [10], "tuning_learning_rates": [.001],
        "scaling_sizes": [2, 4], "study_seeds": [321, 777], "tuning_seeds": [123],
        "point_chunk": 64, "checkpoint_every": 1, "publish_dpi": 90,
        "save_pdf_figures": False, "composed_plot_cases_per_family": 1,
    }
    config_file = workspace / "smoke.json"
    config_file.write_text(json.dumps(config, indent=2))
    commands = [["generate"], ["generate-ood"], ["plot-sampling"]]
    for stage in ("grid", "grid_gelu", "lr"):
        commands += [["tune", "--stage-name", stage], ["tune-select", "--stage-name", stage]]
    commands += [["train-scaling"], ["scaling-report"], ["summarize", "--train-size", "4"],
                 ["plot", "--run-name", "scale_deeponet_N0004_seed321"],
                 ["evaluate-ood", "--run-name", "scale_deeponet_N0004_seed321"]]
    for command in commands:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "run.py"), *command, "--config", str(config_file)], check=True, cwd=PROJECT_ROOT)
    print(f"Smoke test passed. Artifacts: {workspace}")


def dispatch(args, cfg):
    if args.stage == "rename-outputs":
        from src.analysis.output_names import rename_outputs
        return rename_outputs(cfg, dry_run=args.dry_run)
    if args.stage == "generate":
        from src.data.generate import generate_selected_simple_geometry_datasets
        generated = generate_selected_simple_geometry_datasets(cfg)
        from src.data.compose import compose_training_geometry_dataset
        compose_training_geometry_dataset(cfg)
        return generated
    if args.stage == "compose":
        from src.data.compose import compose_training_geometry_dataset
        return compose_training_geometry_dataset(cfg)
    if args.stage in ("train", "train-final"):
        from src.training.trainer import train_model
        from src.analysis.benchmark import hardware_report
        if args.stage == "train-final":
            from src.training.tuning import final_settings
            cfg = final_settings(cfg, cfg.training_seed)
            if args.run_name:
                cfg.run_name = args.run_name
        result = train_model(cfg)
        (Path(cfg.log_outdir) / "hardware.json").write_text(json.dumps(hardware_report(), indent=2))
        return result
    if args.stage == "evaluate":
        from src.analysis.metrics import evaluate
        return evaluate(cfg, split=args.split)
    if args.stage == "generate-ood":
        from src.data.external import generate_selected_ood_datasets
        return generate_selected_ood_datasets(cfg)
    if args.stage == "evaluate-ood":
        from src.analysis.metrics import evaluate
        return evaluate(cfg, external=True, make_plots=cfg.composed_plot_cases_per_family > 0)
    if args.stage == "preview":
        from src.analysis.preview import preview_stored_geometry
        return preview_stored_geometry(cfg)
    if args.stage == "export-ansys":
        from src.analysis.export_ansys import export_stored_geometry_for_ansys
        return export_stored_geometry_for_ansys(cfg)
    if args.stage == "plot-sampling":
        from src.analysis.sampling import plot_sampling
        return plot_sampling(cfg)
    if args.stage == "mesh-check":
        from src.analysis.mesh_check import run_mesh_check
        return run_mesh_check(cfg)
    if args.stage == "plot":
        from src.analysis.metrics import plot_loss, evaluate
        plot_loss(cfg)
        return evaluate(cfg, split=args.split, selected_only=True, make_plots=True)
    if args.stage == "benchmark":
        from src.analysis.benchmark import benchmark_model
        return benchmark_model(cfg, repeats=args.benchmark_repeats, epoch_repeats=args.benchmark_epochs, warmups=args.warmups)
    if args.stage == "report-tuning":
        from src.analysis.tuning_report import tuning_report
        seeds = number_list(args.seeds) if args.seeds else cfg.tuning_seeds
        return tuning_report(cfg, args.stage_name or "grid", seeds, phase=args.report_phase,
                             model_seed=args.report_seed, directory=args.report_dir)
    if args.stage in ("tune", "tune-select"):
        from src.training.tuning import run_stage, select_stage
        seeds = number_list(args.seeds) if args.seeds else cfg.tuning_seeds
        stage = args.stage_name or "grid"
        if args.stage == "tune-select":
            return select_stage(cfg, stage, seeds)
        cell = args.cell
        if cell is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            cell = int(os.environ["SLURM_ARRAY_TASK_ID"])
        return run_stage(cfg, stage, seeds, index=cell)
    if args.stage == "train-scaling":
        from src.training.scaling_runs import train_scaling_run
        seeds = number_list(args.seeds) if args.seeds else cfg.study_seeds
        runs = [(cfg.train_size, cfg.training_seed)] if cfg.train_size else [(size, seed) for size in cfg.scaling_sizes for seed in seeds]
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
            if not 0 <= index < len(runs):
                raise ValueError("Slurm array index exceeds the size/seed combinations.")
            runs = [runs[index]]
        for size, seed in runs:
            train_scaling_run(cfg, size, seed)
        return
    if args.stage in ("summarize", "compare", "scaling-report"):
        from src.analysis.report import summarize, scaling_report
        seeds = number_list(args.seeds) if args.seeds else cfg.study_seeds
        if args.stage == "scaling-report":
            return scaling_report(cfg, cfg.scaling_sizes, seeds)
        return summarize(cfg, cfg.train_size or max(cfg.scaling_sizes), seeds)
    if args.stage == "gpu-info":
        from src.training.runtime import configure_device
        from src.analysis.benchmark import hardware_report
        configure_device(cfg)
        print(json.dumps(hardware_report(), indent=2))
    elif args.stage == "config":
        print(json.dumps(asdict(cfg), indent=2))
    elif args.stage == "smoke":
        smoke()


def main():
    args = arguments()
    cfg = settings_from_arguments(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OMP_NUM_THREADS"] = str(cfg.cpu_threads)
    if not args.show:
        os.environ["MPLBACKEND"] = "Agg"
    return dispatch(args, cfg)


if __name__ == "__main__":
    main()
