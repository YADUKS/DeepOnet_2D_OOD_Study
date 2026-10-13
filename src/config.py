"""
Central settings for the NGSolve + DeepONet workflow.

Active study: nine simple families, independent LHS holdouts, Halton training
prefixes, and four separate OOD families.
    - per-family source pools and configurable training contributions
    - full SDF field as the branch input
    - normalized (x, y) and local SDF(x,y) as the trunk input
    - logging of loss histories, metrics, training summary, and figures
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.fem.geometry_spec import OOD_GEOMETRY_FAMILIES, TRAINING_GEOMETRY_FAMILIES

@dataclass
class Settings:
    # ============================================================
    # STUDY SELECTION
    # ============================================================
    # Retained legacy circle/rectangle settings (not used by the active CLI).
    # Choose one: "circle", "rect", "combined"
    dataset_family: str = "combined"

    # Current dataset size to use
    selected_dataset_size: int = 2000

    # Retained legacy batch-generation choices
    generate_dataset_families: tuple = ("circle", "rect", "combined")
    generate_dataset_sizes: tuple = (500, 1000, 2000, 4000)

    # Portable storage roots and runtime choices
    data_root: str = "data"
    results_root: str = "outputs/study_results"
    run_name: str = ""
    device: str = "auto"
    cpu_threads: int = 4
    force_generation: bool = False
    stored_preview_save_individual: bool = False
    validation_cases_per_family: dict = field(default_factory=lambda: {
        family: 50 for family in TRAINING_GEOMETRY_FAMILIES
    })
    test_cases_per_family: dict = field(default_factory=lambda: {
        family: 50 for family in TRAINING_GEOMETRY_FAMILIES
    })

    # Independently reusable per-family sources and separate OOD datasets.
    training_geometry_families: tuple = TRAINING_GEOMETRY_FAMILIES
    ood_geometry_families: tuple = OOD_GEOMETRY_FAMILIES
    geometry_dataset_root_name: str = "geometry_datasets"
    ood_dataset_root_name: str = "ood_datasets"
    generate_ood_families: tuple = OOD_GEOMETRY_FAMILIES
    ood_cases_per_family: dict = field(default_factory=lambda: {
        "star": 20,
        "aircraft": 20,
        "leaf": 20,
        "fin_like": 20,
    })
    ood_candidate_multiplier: float = 2.5
    # Select the OOD geometry families to evaluate.
    # Available: "star", "aircraft", "leaf", "fin_like"

    ood_evaluation_source_sizes: dict = field(default_factory=lambda: {
        "star": 20,
        "aircraft": 20,
        "leaf": 20,
        "fin_like": 20,
    })
    ood_evaluation_batch_size: int = 8
    # Select exact zero-based cases from each selected OOD family dataset.
    # Use {} to evaluate every case from ood_evaluation_families.

    ood_evaluation_families: tuple = OOD_GEOMETRY_FAMILIES

    ood_test_case_numbers: dict = field(default_factory=dict)

    ood_plot_case_numbers: tuple = (0, 1, 2)
    ood_evaluation_folder_name: str = "ood_evaluation"
    ood_save_pdf_figures: bool = True

    # Held-out composed-simple evaluation
    # Select the simple geometry families to evaluate.
    # Available: "circle", "ellipse", "rectangle", "triangle", "rhombus",
    # "trapezoid", "regular_polygon", "t_shape", "cross_shape"

    # Zero-based positions within each family's held-out test set.
    # Use {} to evaluate all held-out cases from the selected families.
    composed_evaluation_families: tuple = TRAINING_GEOMETRY_FAMILIES

    composed_test_case_numbers: dict = field(default_factory=dict)

    composed_plot_cases_per_family: int = 3
    composed_evaluation_folder_name: str = "composed_test_evaluation"
    composed_save_pdf_figures: bool = True

    # Stored dataset-case preview (no generation or prediction)
    preview_dataset_group: str = "training"  # "training" or "ood"
    preview_dataset_family: str = "circle"
    preview_dataset_size: int = 500
    preview_dataset_case_number: int = 0
    stored_preview_folder_name: str = "geometry_previews"
    stored_preview_save_pdf: bool = True
    stored_preview_show: bool = False

    # Exact stored-case export for ANSYS (training, composed, or OOD).
    ansys_export_dataset_group: str = "composed"
    # For composed exports, select this family first. case_number is then the
    # zero-based position within this family, not within the mixed pool.
    ansys_export_family: str = "circle"
    ansys_export_dataset_size: int = 500
    ansys_export_case_number: int = 0
    ansys_export_composition_id: str = ""
    ansys_extrusion_thickness: float = 0.001
    ansys_export_units: str = "mm"
    ansys_export_folder_name: str = "ansys_exports"

    generate_geometry_families: tuple = TRAINING_GEOMETRY_FAMILIES
    geometry_cases_per_family: dict = field(default_factory=lambda: {family: 800 for family in TRAINING_GEOMETRY_FAMILIES})
    geometry_candidate_multiplier: float = 2.0

    # Select independently stored datasets for a composed training pool.
    # source_sizes chooses which generated file to load; counts chooses how
    # many deterministic cases from that file to include.  Zero excludes a
    # family without deleting its reusable source dataset.
    training_geometry_source_sizes: dict = field(default_factory=lambda: {family: 800 for family in TRAINING_GEOMETRY_FAMILIES})
    training_geometry_counts: dict = field(default_factory=lambda: {family: 800 for family in TRAINING_GEOMETRY_FAMILIES})
    composition_seed: int = 24680  # Retained composition tag; no membership shuffling
    composed_dataset_root_name: str = "composed_datasets"

    # "legacy" keeps the original circle/rectangle path.  "composed" is the
    # new selectable multi-family design-space workflow.
    training_dataset_mode: str = "composed"
    training_composition_id: str = ""

    # ============================================================
    # FILE NAMES
    # ============================================================
    dataset_name: str = "dataset.npz"
    model_name: str = "best_model.pt"

    train_loss_csv_name: str = "train_loss_history.csv"
    val_loss_csv_name: str = "val_loss_history.csv"
    training_summary_name: str = "training_summary.txt"
    metrics_csv_name: str = "test_metrics.csv"
    split_indices_name: str = "split_indices.npz"
    config_used_name: str = "config_used.json"
    temperature_normalization_name: str = "temperature_normalization.npz"

    loss_plot_name: str = "train_val_loss.png"
    combined_figure_name: str = "test_case_comparison_combined.png"

    # ============================================================
    # GEOMETRY
    # ============================================================
    Lx: float = 0.036
    Ly: float = 0.021

    # Fixed centre (used only when random_position = False)
    cx_fixed: float = 0.018
    cy_fixed: float = 0.0105
    random_position: bool = True

    # Shared design-space controls for all new geometry families.
    geometry_rotation_min: float = 0.0
    geometry_rotation_max: float = 6.283185307179586
    geometry_minimum_clearance: float = 0.0016
    geometry_minimum_area: float = 1.0e-5
    geometry_minimum_edge_length: float = 0.00005
    curved_boundary_segments: int = 96
    geometry_seed_offsets: dict = field(default_factory=lambda: {
        "circle": 0,
        "ellipse": 10000,
        "rectangle": 20000,
        "triangle": 30000,
        "rhombus": 40000,
        "trapezoid": 50000,
        "regular_polygon": 60000,
        "t_shape": 70000,
        "cross_shape": 80000,
        "star": 100000,
        "aircraft": 110000,
        "leaf": 120000,
        "fin_like": 130000,
    })

    # OOD star ranges
    star_points: tuple = (5, 6, 7, 8)
    star_outer_radius_min: float = 0.0045
    star_outer_radius_max: float = 0.007
    star_inner_ratio_min: float = 0.38
    star_inner_ratio_max: float = 0.62

    # OOD aircraft ranges
    aircraft_length_min: float = 0.010
    aircraft_length_max: float = 0.015
    aircraft_wingspan_min: float = 0.009
    aircraft_wingspan_max: float = 0.015
    aircraft_fuselage_width_ratio_min: float = 0.12
    aircraft_fuselage_width_ratio_max: float = 0.22
    aircraft_tailspan_ratio_min: float = 0.35
    aircraft_tailspan_ratio_max: float = 0.55
    aircraft_wing_sweep_min: float = 0.05
    aircraft_wing_sweep_max: float = 0.25

    # OOD leaf ranges
    leaf_length_min: float = 0.010
    leaf_length_max: float = 0.015
    leaf_width_min: float = 0.005
    leaf_width_max: float = 0.010
    leaf_curvature_min: float = -0.25
    leaf_curvature_max: float = 0.25
    leaf_sharpness_min: float = 0.80
    leaf_sharpness_max: float = 1.50

    # OOD connected engineering fin/comb ranges
    fin_height_min: float = 0.010
    fin_height_max: float = 0.015
    fin_spine_width_min: float = 0.0015
    fin_spine_width_max: float = 0.0030
    fin_length_min: float = 0.006
    fin_length_max: float = 0.012
    fin_counts: tuple = (4, 5, 6, 7, 8)
    fin_gap_ratio_min: float = 0.60
    fin_gap_ratio_max: float = 1.60
    fin_taper_ratio_min: float = 0.70
    fin_taper_ratio_max: float = 1.00

    # Circle diameter range
    D_min: float = 0.008
    D_max: float = 0.014

    # Rectangle width / height range
    W_min: float = 0.008
    W_max: float = 0.014
    H_min: float = 0.006
    H_max: float = 0.012

    # Ellipse full-axis ranges
    ellipse_major_axis_min: float = 0.009
    ellipse_major_axis_max: float = 0.014
    ellipse_aspect_ratio_min: float = 1.15
    ellipse_aspect_ratio_max: float = 2.25

    # Triangle ranges; skew is apex displacement divided by base length
    triangle_base_min: float = 0.008
    triangle_base_max: float = 0.014
    triangle_height_min: float = 0.007
    triangle_height_max: float = 0.013
    triangle_skew_min: float = -0.30
    triangle_skew_max: float = 0.30

    # Rhombus diagonal ranges
    rhombus_diagonal_x_min: float = 0.008
    rhombus_diagonal_x_max: float = 0.014
    rhombus_diagonal_y_min: float = 0.006
    rhombus_diagonal_y_max: float = 0.012

    # Trapezoid ranges
    trapezoid_bottom_width_min: float = 0.009
    trapezoid_bottom_width_max: float = 0.014
    trapezoid_top_ratio_min: float = 0.45
    trapezoid_top_ratio_max: float = 0.90
    trapezoid_height_min: float = 0.006
    trapezoid_height_max: float = 0.012
    trapezoid_top_offset_fraction_min: float = -0.20
    trapezoid_top_offset_fraction_max: float = 0.20

    # Regular polygons use 5--10 sides because triangles and four-sided
    # families are represented separately.
    regular_polygon_sides: tuple = (5, 6, 7, 8, 9, 10)
    regular_polygon_radius_min: float = 0.0045
    regular_polygon_radius_max: float = 0.007

    # T-shape ranges. The ratios keep the stem connected to, and narrower
    # than, the top bar while allowing size, aspect ratio and arm variation.
    t_shape_top_width_min: float = 0.009
    t_shape_top_width_max: float = 0.014
    t_shape_height_min: float = 0.008
    t_shape_height_max: float = 0.013
    t_shape_stem_width_ratio_min: float = 0.20
    t_shape_stem_width_ratio_max: float = 0.50
    t_shape_top_thickness_ratio_min: float = 0.18
    t_shape_top_thickness_ratio_max: float = 0.42

    # Cross-shape ranges. Width and height control overall aspect ratio; the
    # two ratios independently control vertical- and horizontal-arm thickness.
    cross_shape_width_min: float = 0.009
    cross_shape_width_max: float = 0.014
    cross_shape_height_min: float = 0.008
    cross_shape_height_max: float = 0.013
    cross_shape_vertical_width_ratio_min: float = 0.20
    cross_shape_vertical_width_ratio_max: float = 0.50
    cross_shape_horizontal_thickness_ratio_min: float = 0.20
    cross_shape_horizontal_thickness_ratio_max: float = 0.50

    # ============================================================
    # EXTERNAL TEST GEOMETRY: ROUNDED RECTANGLE
    # ============================================================
    rounded_test_n_cases: int = 10
    rounded_test_case_number: int = 8
    rounded_test_seed: int = 777
    rounded_test_dataset_name: str = "rounded_rect_test_dataset.npz"
    rounded_test_metrics_csv_name: str = "rounded_rect_test_metrics.csv"
    rounded_test_summary_name: str = "rounded_rect_test_summary.txt"
    rounded_test_folder_name: str = "rounded_rect_external_test"

    rounded_W_min: float = 0.008
    rounded_W_max: float = 0.014
    rounded_H_min: float = 0.006
    rounded_H_max: float = 0.012
    rounded_R_min: float = 0.0005
    rounded_R_max: float = 0.003
    rounded_margin: float = 0.003
    rounded_corner_segments: int = 12

    # ============================================================
    # PHYSICS
    # ============================================================
    kappa: float = 28.3
    h_hole: float = 4000.0
    T_inf: float = 300.0
    q_top: float = 0.5e6

    # ============================================================
    # FE SETTINGS
    # ============================================================
    order: int = 2
    maxh: float = 0.0008

    # ============================================================
    # CARTESIAN GRID FOR DEEPONET
    # ============================================================
    Nx: int = 128
    Ny: int = 96

    # ============================================================
    # DATASET GENERATION
    # ============================================================
    n_cases: int = 2000
    seed: int = 42

    # ============================================================
    # PREVIEW CASE
    # ============================================================
    preview_shape: str = "circle"  # "circle" or "rect"
    preview_D: float = 0.012
    preview_W: float = 0.012
    preview_H: float = 0.008
    preview_cx: float = 0.018
    preview_cy: float = 0.0105

    # ============================================================
    # LEGACY FRACTIONAL SPLITS (active v2 uses explicit per-family counts)
    # ============================================================
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    split_seed: int = 123

    # ============================================================
    # ML SETTINGS
    # ============================================================
    test_case_number: int = 0
    epochs: int = 5000
    batch_size: int = 8
    lr: float = 1e-3
    latent_dim: int = 128
    operator: str = "deeponet"
    variant: str = "custom"
    branch_hidden: tuple = (256, 256)
    trunk_hidden: tuple = (128, 128)
    deeponet_activation: str = "tanh"
    train_size: int = 0  # PER FAMILY, zero means the complete selected pool
    tuning_size: int = 200  # PER FAMILY; selected once, before scaling runs
    scaling_sizes: tuple = (50, 100, 200, 400, 800)
    study_seeds: tuple = (321, 777, 2026, 31415, 27182)
    tuning_seeds: tuple = (123,)
    tuning_layers: tuple = (2, 4, 6, 8, 10)
    tuning_neurons: tuple = (10, 20, 40, 80, 160)
    tuning_learning_rates: tuple = (3e-4, 1e-3, 3e-3)
    tuning_epochs: int = 2000
    precision: str = "fp32"
    point_chunk: int = 2048
    grad_clip: float = 1.0
    checkpoint_every: int = 50
    resume: bool = False
    custom_geometry_ranges: dict = field(default_factory=dict)
    training_seed: int = 123
    deterministic_training: bool = True
    early_stop_patience: int = 800
    scheduler_patience: int = 200
    scheduler_factor: float = 0.5

    # ============================================================
    # PLOTTING / EXPORT
    # ============================================================
    save_figures: bool = True
    show_figures: bool = False
    save_loss_plot: bool = True
    show_loss_plot: bool = False
    export_vtk: bool = False

    # High-quality figure settings
    publish_dpi: int = 600
    save_pdf_figures: bool = True

    # ============================================================
    # PROPERTIES
    # ============================================================
    @property
    def output_root(self):
        path = Path(self.results_root)
        return path.parent if path.name == "study_results" else path

    @property
    def study_dir(self):
        return self.output_root / "study_results" / self.case_tag

    @property
    def checkpoint_outdir(self):
        return str(self.output_root / "checkpoints" / self.case_tag / self.run_name)

    @property
    def log_outdir(self):
        return str(self.output_root / "logs" / self.case_tag / self.run_name)

    @property
    def figure_outdir(self):
        return str(self.output_root / "figures" / self.case_tag / self.run_name)

    def ensure_directories(self):
        for path in (self.outdir, self.checkpoint_outdir, self.log_outdir, self.figure_outdir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        from src.fem.registry import TRAINING_SHAPES
        for family, spec in TRAINING_SHAPES.items():
            self.geometry_seed_offsets.setdefault(family, int(spec["seed_offset"]))
        if len(set(self.geometry_seed_offsets.values())) != len(self.geometry_seed_offsets):
            raise ValueError("Geometry seed offsets must be distinct.")
        for name in ("data_root", "results_root"):
            path = Path(getattr(self, name)).expanduser()
            setattr(self, name, str(path if path.is_absolute() else PROJECT_ROOT / path))
        if self.run_name and (self.run_name in (".", "..") or any(c in self.run_name for c in "/\\")):
            raise ValueError("run_name must be a single directory name.")

    @property
    def case_tag(self) -> str:
        if str(self.training_dataset_mode).strip().lower() == "composed":
            from src.data.compose import get_composition_id

            composition_id = (
                str(self.training_composition_id).strip()
                or get_composition_id(self)
            )
            return f"design_space_{composition_id}"
        return f"{self.dataset_family}_{self.selected_dataset_size:04d}"

    @property
    def outdir(self) -> str:
        return str(self.study_dir / self.run_name)

    @property
    def geometry_datasets_root(self) -> str:
        return os.path.join(self.data_root, self.geometry_dataset_root_name)

    @property
    def ood_datasets_root(self) -> str:
        return os.path.join(self.data_root, self.ood_dataset_root_name)

    @property
    def composed_datasets_root(self) -> str:
        return os.path.join(self.data_root, self.composed_dataset_root_name)

    def geometry_family_dir(self, family: str) -> str:
        return os.path.join(self.geometry_datasets_root, str(family))

    def geometry_dataset_dir(self, family: str, n_cases: int) -> str:
        from src.data.design import design_id
        return os.path.join(self.geometry_family_dir(family),
                            f"design_{design_id(self, family)}",
                            f"train_{int(n_cases):06d}_val_{int(self.validation_cases_per_family[family]):06d}_test_{int(self.test_cases_per_family[family]):06d}")

    def geometry_dataset_path(self, family: str, n_cases: int) -> str:
        return os.path.join(
            self.geometry_dataset_dir(family, n_cases),
            self.dataset_name,
        )

    def geometry_fe_dir(self, family: str, n_cases: int) -> str:
        return os.path.join(
            self.geometry_dataset_dir(family, n_cases),
            "fe_cases",
        )

    def ood_family_dir(self, family: str) -> str:
        return os.path.join(self.ood_datasets_root, str(family))

    def ood_dataset_dir(self, family: str, n_cases: int) -> str:
        from src.data.design import design_id
        return os.path.join(self.ood_family_dir(family), f"design_{design_id(self, family, ood=True)}",
                            f"cases_{int(n_cases):06d}")

    def ood_dataset_path(self, family: str, n_cases: int) -> str:
        return os.path.join(
            self.ood_dataset_dir(family, n_cases),
            self.dataset_name,
        )

    def ood_fe_dir(self, family: str, n_cases: int) -> str:
        return os.path.join(
            self.ood_dataset_dir(family, n_cases),
            "fe_cases",
        )

    def composed_dataset_dir(self, composition_id: str) -> str:
        return os.path.join(
            self.composed_datasets_root,
            f"mix_{composition_id}",
        )

    def composed_dataset_path(self, composition_id: str) -> str:
        return os.path.join(
            self.composed_dataset_dir(composition_id),
            self.dataset_name,
        )

    @property
    def fe_outdir(self) -> str:
        return os.path.join(self.outdir, "fe_cases")

    @property
    def dataset_path(self) -> str:
        if self.training_dataset_mode == "composed":
            from src.data.compose import get_composition_id
            return self.composed_dataset_path(self.training_composition_id or get_composition_id(self))
        return os.path.join(self.outdir, self.dataset_name)

    @property
    def model_path(self) -> str:
        return str(Path(self.checkpoint_outdir) / self.model_name)

    @property
    def split_indices_path(self) -> str:
        return os.path.join(self.outdir, self.split_indices_name)

    @property
    def config_used_path(self) -> str:
        return os.path.join(self.outdir, self.config_used_name)

    @property
    def temperature_normalization_path(self) -> str:
        return os.path.join(self.outdir, self.temperature_normalization_name)

    @property
    def ood_evaluation_root(self) -> str:
        return os.path.join(self.outdir, self.ood_evaluation_folder_name)

    def ood_evaluation_family_dir(self, family: str) -> str:
        return os.path.join(self.ood_evaluation_root, str(family))

    @property
    def composed_evaluation_root(self) -> str:
        return os.path.join(self.outdir, self.composed_evaluation_folder_name)

    def composed_evaluation_family_dir(self, family: str) -> str:
        return os.path.join(self.composed_evaluation_root, str(family))

    @property
    def stored_preview_root(self) -> str:
        return os.path.join(self.results_root, self.stored_preview_folder_name)

    def stored_preview_output_dir(
        self,
        group: str,
        family: str,
        dataset_size: int,
        case_number: int,
    ) -> str:
        return os.path.join(
            self.stored_preview_root,
            str(group),
            str(family),
            f"cases_{int(dataset_size):06d}",
            f"case_{int(case_number):06d}",
        )

    @property
    def ansys_export_root(self) -> str:
        return os.path.join(self.results_root, self.ansys_export_folder_name)

    def ansys_export_output_dir(
        self,
        group: str,
        family: str,
        dataset_size: int,
        case_number: int,
        composition_id=None,
    ) -> str:
        dataset_tag = (
            f"mix_{composition_id}"
            if str(group) == "composed"
            else f"cases_{int(dataset_size):06d}"
        )
        return os.path.join(
            self.ansys_export_root,
            str(group),
            str(family),
            dataset_tag,
            f"case_{int(case_number):06d}",
        )

    @property
    def train_loss_csv_path(self) -> str:
        return os.path.join(self.outdir, self.train_loss_csv_name)

    @property
    def val_loss_csv_path(self) -> str:
        return os.path.join(self.outdir, self.val_loss_csv_name)

    @property
    def training_summary_path(self) -> str:
        return os.path.join(self.outdir, self.training_summary_name)

    @property
    def metrics_csv_path(self) -> str:
        return os.path.join(self.outdir, self.metrics_csv_name)

    @property
    def loss_plot_path(self) -> str:
        return os.path.join(self.outdir, self.loss_plot_name)

    @property
    def combined_figure_path(self) -> str:
        return os.path.join(self.outdir, self.combined_figure_name)

    @property
    def rounded_test_outdir(self) -> str:
        return os.path.join(self.outdir, self.rounded_test_folder_name)

    @property
    def rounded_test_fe_outdir(self) -> str:
        return os.path.join(self.rounded_test_outdir, "fe_cases")

    @property
    def rounded_test_dataset_path(self) -> str:
        return os.path.join(self.rounded_test_outdir, self.rounded_test_dataset_name)

    @property
    def rounded_test_metrics_csv_path(self) -> str:
        return os.path.join(self.rounded_test_outdir, self.rounded_test_metrics_csv_name)

    @property
    def rounded_test_summary_path(self) -> str:
        return os.path.join(self.rounded_test_outdir, self.rounded_test_summary_name)
