# SDF-in-trunk DeepONet geometry design-space study

This project trains an NGSolve/DeepONet model on varied **simple hollow
geometries** and evaluates its generalization to completely separate
**out-of-distribution (OOD) hollow geometries**.

The new workflow is deterministic and keeps every geometry family in an
independent reusable dataset.

## Fixed DeepONet inputs

The model contract is not changed by geometry family:

```text
Branch input:
    complete flattened SDF field
    shape = [batch, Nx * Ny]

Trunk input:
    [x_normalized, y_normalized, sdf(x,y)]
    shape = [batch, Nx * Ny, 3]
```

The Cartesian coordinates are normalized by the physical domain dimensions:

```text
x_normalized = x / Lx
y_normalized = y / Ly
```

The SDF convention is:

```text
SDF < 0: hollow region
SDF = 0: hollow boundary
SDF > 0: solid material
```

## Geometry groups

### Simple training families

- Circle
- Ellipse
- Rectangle
- Triangle
- Rhombus
- Trapezoid
- Regular polygon with 5--10 sides

Each family varies in size, position, rotation, aspect ratio where applicable,
and family-specific shape parameters.

### OOD evaluation families

- Star
- Aircraft silhouette
- Leaf
- Engineering fin-like geometry

The engineering fin is one connected comb-shaped boundary consisting of a
vertical spine and 4--8 attached horizontal rectangular fins.

OOD families are marked `training_eligible=False`, are stored under a separate
root, and are rejected by the training composer.

## Main files

| File | Purpose |
|---|---|
| `config.py` | All user selections, parameter ranges, seeds, and paths |
| `geometry_spec.py` | Shared geometry case, transformations, validation, polygon SDF |
| `simple_geometries.py` | Seven simple training-family boundary constructors |
| `ood_geometries.py` | Star, aircraft, leaf, and engineering-fin constructors |
| `geometry.py` | NGSolve mesh construction and shared mesh/SDF entry points |
| `solver.py` | NGSolve thermal finite-element solver |
| `sampling.py` | Cartesian sampling of FE solutions |
| `fe_storage.py` | Per-case FE mesh, temperature, boundary, and metadata storage |
| `generate_geometry_datasets.py` | Independent simple-family generation |
| `compose_geometry_dataset.py` | Deterministic family selection and composition |
| `training_data.py` | Validation, stratified splitting, and training-only normalization |
| `ml_dataset.py` | PyTorch datasets and explicit split-aware loaders |
| `model.py` | SDF-branch and three-feature-trunk DeepONet |
| `train_deeponet.py` | Deterministic model training and reproducibility artifacts |
| `evaluate_composed.py` | Exact held-out simple-family prediction and FE errors |
| `generate_ood_datasets.py` | Independent OOD-family generation |
| `evaluate_ood.py` | OOD inference, metrics, plots, and provenance |
| `preview_stored_geometry.py` | Preview one actual stored training or OOD case |
| `export_ansys_geometry.py` | Export an exact stored training, composed, or OOD case for ANSYS |
| `main_pipeline.py` | Ordered and guarded workflow controller |

## Recommended workflow

Use `main_pipeline.py` for every stage. Keep all switches `False` except the
stage or consecutive stages intentionally being run.

Run from this project directory with the Python environment containing
NGSolve, Netgen, PyTorch, NumPy, and Matplotlib:

```powershell
python main_pipeline.py
```

On Windows, the project virtual environment may be invoked directly:

```powershell
& "..\.venv\Scripts\python.exe" main_pipeline.py
```

### Preview one actual stored geometry

The preview tool reads an existing dataset case. It does not create a new
geometry, run a new FE solve, train a model, or perform prediction.

For a simple training-family case:

```python
preview_dataset_group = "training"
preview_dataset_family = "triangle"
preview_dataset_size = 500
preview_dataset_case_number = 12
```

For an OOD case:

```python
preview_dataset_group = "ood"
preview_dataset_family = "fin_like"
preview_dataset_size = 100
preview_dataset_case_number = 5
```

Enable:

```python
RUN_PREVIEW_STORED_GEOMETRY = True
```

The combined preview contains the original hollow boundary, original NGSolve
FE mesh, stored SDF, stored mask, stored Cartesian physical temperature, and
original FE temperature contour.

When this stage starts, it asks:

```text
Save the six preview panels as individual figures? [y/n]:
```

Enter `y` to save all six panels separately in an
`individual_figures` subfolder. Enter `n` to save only the combined preview.
If `stored_preview_save_pdf = True`, each selected individual panel is saved
as both PNG and PDF; otherwise, only PNG files are saved.

Output:

```text
study_results/
    geometry_previews/
        training_or_ood/
            family/
                cases_<size>/
                    case_<number>/
                        preview_combined.png
                        preview_combined.pdf
                        preview_metadata.json
                        individual_figures/
                            01_original_geometry.png
                            02_fe_mesh.png
                            03_sdf.png
                            04_mask.png
                            05_cartesian_temperature.png
                            06_fe_temperature.png
```

### Export one stored case for ANSYS

The ANSYS exporter extrudes the exact stored NGSolve plate mesh. It supports
independent simple-family datasets, a composed training pool, and OOD
datasets. It does not regenerate the geometry.

For an OOD case:

```python
ansys_export_dataset_group = "ood"
ansys_export_family = "star"
ansys_export_dataset_size = 20
ansys_export_case_number = 0
ansys_extrusion_thickness = 0.001  # metres
ansys_export_units = "mm"
```

For an independent simple-family case, use `"training"` and select the
family and stored dataset size:

```python
ansys_export_dataset_group = "training"
ansys_export_family = "triangle"
ansys_export_dataset_size = 500
ansys_export_case_number = 0
```

For an exact case from the composed pool:

```python
ansys_export_dataset_group = "composed"
ansys_export_family = "circle"
ansys_export_case_number = 0
ansys_export_composition_id = ""  # derive from current composition settings
```

For a composed export, the family is selected first and the case number is
the zero-based position within that family's cases, not within the shuffled
mixed pool. `ansys_export_dataset_size` is ignored for composed datasets.

Set `ansys_export_composition_id` explicitly to export an older saved mix
whose ID differs from the current composition settings. A composed dataset
must already have been created before it can be exported.

Enable only:

```python
RUN_EXPORT_STORED_GEOMETRY_FOR_ANSYS = True
```

The output contains a watertight, extruded ASCII STL; exact hollow-boundary
coordinates; classified boundary edges; geometry and physics metadata;
ANSYS boundary-condition notes; and a mesh preview:

```text
study_results/
    ansys_exports/
        training_composed_or_ood/
            family/
                cases_<size>_or_mix_<composition_id>/
                    case_<number>/
                        plate_with_hollow_cutout.stl
                        boundary_edges.csv
                        hollow_boundary.csv
                        geometry_parameters.json
                        ansys_boundary_conditions.txt
                        geometry_preview.png
```

STL does not store units or named selections. The configured coordinate units
and ANSYS face mapping are therefore recorded in the JSON, CSV, and text
files. For two-dimensional equivalence, the front and back extrusion faces
must be adiabatic in ANSYS.

### Step 1: generate separate simple-family datasets

Select families and independent sizes in `config.py`:

```python
generate_geometry_families = (
    "circle",
    "ellipse",
    "rectangle",
    "triangle",
    "rhombus",
    "trapezoid",
    "regular_polygon",
    "t_shape",
    "cross_shape",
)

geometry_cases_per_family = {
    "circle": 500,
    "ellipse": 500,
    "rectangle": 500,
    "triangle": 500,
    "rhombus": 500,
    "trapezoid": 500,
    "regular_polygon": 500,
    "t_shape": 500,
    "cross_shape": 500,
}
```

Enable:

```python
RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS = True
```

Output example:

```text
study_results/
    geometry_datasets/
        circle/
            cases_000500/
                dataset.npz
                manifest.json
                fe_cases/
        ellipse/
        rectangle/
        triangle/
        rhombus/
        trapezoid/
        regular_polygon/
        t_shape/
        cross_shape/
```

Generating another size does not overwrite an existing size.

### Step 2: compose the training pool

Choose which saved source file to use for every family:

```python
training_geometry_source_sizes = {
    "circle": 500,
    "ellipse": 500,
    "rectangle": 500,
    "triangle": 500,
    "rhombus": 500,
    "trapezoid": 500,
    "regular_polygon": 500,
    "t_shape": 500,
    "cross_shape": 500,
}
```

Choose the exact contribution to the training pool:

```python
training_geometry_counts = {
    "circle": 500,
    "ellipse": 500,
    "rectangle": 250,
    "triangle": 250,
    "rhombus": 0,
    "trapezoid": 250,
    "regular_polygon": 500,
    "t_shape": 0,
    "cross_shape": 0,
}
```

Zero excludes a family without deleting its stored data. A selected count
cannot exceed its configured source size.

For example, to add only T shapes to the current pool, set `t_shape` to a
positive count and leave `cross_shape` at zero. To use both, give both
families positive counts. Each choice produces a distinct composition ID,
split, result directory, and trained model while retaining the independent
source datasets for later experiments.

Enable:

```python
RUN_COMPOSE_TRAINING_GEOMETRY_DATASET = True
```

The deterministic composition is stored under:

```text
study_results/composed_datasets/mix_<composition-ID>/
```

It contains physical temperature values. Final normalization is deliberately
not fitted during composition.

### Step 3: train the composed model

Ensure:

```python
training_dataset_mode = "composed"
```

Configure splitting and training:

```python
train_fraction = 0.8
val_fraction = 0.1
split_seed = 123

training_seed = 12345
deterministic_training = True

epochs = 20000
batch_size = 8
lr = 1e-3
latent_dim = 128
```

Enable:

```python
RUN_TRAIN_COMPOSED_MODEL = True
```

Splitting is deterministic and stratified within every selected family.
Temperature `T_min` and `T_max` are fitted from **training cases and solid
points only**. The same unchanged scaler is applied to validation and testing.
The saved normalization method is recorded as `training_solid_minmax`.

Training output:

```text
study_results/
    design_space_<composition-ID>/
        best_deeponet.pt
        split_indices.npz
        temperature_normalization.npz
        config_used.json
        training_summary.txt
        train_loss_history.csv
        val_loss_history.csv
        train_val_loss.png
```

### Step 4: evaluate unseen simple-family test cases

Enable:

```python
RUN_EVALUATE_COMPOSED_TEST_DATASET = True
```

`evaluate_composed.py` loads the exact saved `test_idx` from
`split_indices.npz`. It verifies that training, validation, and test indices
are complete and non-overlapping before prediction.

Select one geometry family and one case using a zero-based position within
that family's held-out test subset:

```python
composed_evaluation_families = ("triangle",)
composed_test_case_numbers = {
    "triangle": (2,),  # third reserved unseen triangle test case
}
```

Multiple cases and families can be selected:

```python
composed_evaluation_families = ("circle", "rectangle")
composed_test_case_numbers = {
    "circle": (0, 4),
    "rectangle": (2,),
}
```

Every explicitly selected case is plotted. To restore the previous behaviour
and evaluate all held-out cases from the selected families, use:

```python
composed_test_case_numbers = {}
```

The evaluator records both the requested family-level test-case number and
its resolved composed case ID in `evaluation_provenance.json`. Invalid,
duplicate, negative, training, or validation selections are rejected.

Results are saved under:

```text
study_results/
    design_space_<composition-ID>/
        composed_test_evaluation/
            all_composed_test_metrics.csv
            composed_test_summary.json
            evaluation_provenance.json
            composed_test_family_error_comparison.png
            circle/
            ellipse/
            rectangle/
            triangle/
            rhombus/
            trapezoid/
            regular_polygon/
            t_shape/
            cross_shape/
```

For selected cases, outputs include Cartesian comparisons plus the original
FE ground truth, prediction interpolated onto the FE mesh, signed FE error,
absolute FE error, relative FE error, separate publication-style figures, and
a case summary text file. Each plotted case is placed in a separate
legacy-style folder:

```text
family/
    test_case_number_XXXX_case_id_YYYYYY/
        test_case_number_XXXX_case_id_YYYYYY_combined.png
        test_case_number_XXXX_case_id_YYYYYY_geometry_with_domain.png
        test_case_number_XXXX_case_id_YYYYYY_fe_mesh.png
        test_case_number_XXXX_case_id_YYYYYY_metrics.csv
        test_case_number_XXXX_case_id_YYYYYY_metadata.json
        case_YYYYYY_comparison.png
        case_YYYYYY_fe_comparison.png
        case_YYYYYY_ground_truth_fe.png
        case_YYYYYY_predicted_fe.png
        case_YYYYYY_signed_error_fe.png
        case_YYYYYY_absolute_error_fe.png
        case_YYYYYY_relative_error_fe.png
        case_YYYYYY_summary.txt
```

The `*_combined` figure follows the legacy 3-by-3 layout: exact geometry,
actual FE mesh, FE truth, FE prediction, Cartesian truth, Cartesian
prediction, signed FE error, absolute FE error, and relative FE error.
The standalone geometry and combined geometry panel use the exact
`boundary_points` stored with the FE case, rather than the pixelated
Cartesian mask.
Each case summary also contains the complete stored geometry record
(family-specific parameters, centre, rotation, scale, case/candidate/seed
identifiers), domain dimensions, source family case ID, FE source path, and
the exact hollow-boundary coordinates in metres. This is sufficient to
reconstruct the evaluated geometry independently.

Control the number of plotted cases per family with:

```python
composed_plot_cases_per_family = 3
```

### Step 5: generate separate OOD datasets

Select OOD families and sizes:

```python
generate_ood_families = (
    "star",
    "aircraft",
    "leaf",
    "fin_like",
)

ood_cases_per_family = {
    "star": 100,
    "aircraft": 100,
    "leaf": 100,
    "fin_like": 100,
}
```

Enable:

```python
RUN_GENERATE_OOD_DATASETS = True
```

Output:

```text
study_results/
    ood_datasets/
        star/
        aircraft/
        leaf/
        fin_like/
```

OOD-local compatibility normalization is stored only for inspection. It must
not be used for model evaluation.

### Step 6: evaluate OOD generalization

Select generated OOD sources:

```python
ood_evaluation_families = (
    "star",
    "aircraft",
    "leaf",
    "fin_like",
)

ood_evaluation_source_sizes = {
    "star": 100,
    "aircraft": 100,
    "leaf": 100,
    "fin_like": 100,
}

ood_evaluation_batch_size = 8
ood_test_case_numbers = {}
ood_plot_case_numbers = (0, 1, 2)
```

To evaluate one exact stored OOD case from one family:

```python
ood_evaluation_families = ("star",)
ood_test_case_numbers = {
    "star": (2,),  # stored star dataset case 2
}
```

To evaluate several exact cases from multiple OOD families:

```python
ood_evaluation_families = ("aircraft", "leaf")
ood_test_case_numbers = {
    "aircraft": (0, 4),
    "leaf": (2,),
}
```

Every explicitly selected OOD case is evaluated and plotted. Case numbers are
zero-based positions in each independently stored OOD family dataset. Invalid,
duplicate, negative, and out-of-range selections are rejected.

Use an empty mapping to evaluate all cases:

```python
ood_test_case_numbers = {}
```

In evaluate-all mode, `ood_plot_case_numbers` controls which cases receive
figures. It is ignored for explicitly selected families because every
explicitly selected case is plotted. Requested and resolved case IDs are
recorded in `evaluation_provenance.json`.

Enable:

```python
RUN_EVALUATE_OOD_DATASETS = True
```

The evaluator loads the model's saved training scaler and ignores each OOD
dataset's local `T_min`/`T_max`. OOD normalized values are not clipped.

Output:

```text
study_results/
    design_space_<composition-ID>/
        ood_evaluation/
            all_ood_metrics.csv
            ood_summary.json
            evaluation_provenance.json
            ood_family_error_comparison.png
            ood_family_error_comparison.pdf
            star/
            aircraft/
            leaf/
            fin_like/
```

Per-case metrics include normalized masked MSE, MAE and RMSE in Kelvin,
relative L2, maximum absolute error, and mean pointwise relative error. The
same metrics are also calculated on the original FE mesh. OOD selected-case
outputs use the same Cartesian and original-style FE figure set as the
composed held-out test evaluator. Every plotted OOD case is also written to
its own `test_case_number_XXXX_case_id_YYYYYY` folder with the same legacy
3-by-3 combined figure, one-row metrics CSV, geometry metadata, individual FE
figures, and summary text.

## Three prediction workflows

```text
1. Legacy prediction:
   evaluate_deeponet.py
   Original circle/rectangle workflow

2. Composed simple-geometry prediction:
   evaluate_composed.py
   Exact saved test_idx from known training families

3. OOD prediction:
   evaluate_ood.py
   Star, aircraft, leaf, and engineering fin-like families
```

The composed and OOD evaluators use the same model checkpoint, branch/trunk
definition, saved training scaler, Cartesian metrics, FE interpolation, FE
metrics, and figure conventions.

## Running several stages together

The new pipeline executes enabled stages in dependency order. A first small
smoke study can therefore use:

```python
RUN_GENERATE_SIMPLE_GEOMETRY_DATASETS = True
RUN_COMPOSE_TRAINING_GEOMETRY_DATASET = True
RUN_TRAIN_COMPOSED_MODEL = True
RUN_EVALUATE_COMPOSED_TEST_DATASET = True
RUN_GENERATE_OOD_DATASETS = True
RUN_EVALUATE_OOD_DATASETS = True
```

For a full expensive study, running one stage at a time is safer because FE
generation and training can take substantial time.

## Determinism

The workflow controls:

- Family-specific deterministic geometry seeds
- Latin hypercube parameter samples
- Dataset subset selection and composition shuffle
- Stratified split assignment
- Python, NumPy, PyTorch, and CUDA seeds
- DataLoader shuffle generator
- Deterministic PyTorch/cuDNN algorithms
- cuBLAS workspace configuration
- TF32 disabling during deterministic CUDA training

Exact floating-point reproducibility still depends on using compatible Python,
library, CUDA, GPU, NGSolve, and Netgen environments. The training snapshot
records those environment details.

## Data integrity rules

Generation and training reject:

- Self-intersecting boundaries
- Shapes outside the domain clearance
- Holes below minimum area or edge resolution
- Invalid SDF/mask signs
- Non-binary masks
- Non-finite model arrays
- Mismatched Cartesian grids
- Mixed FE/physics configurations
- Missing geometry families or source files
- Overlapping or incomplete split indices
- OOD files offered to the training composer

## Legacy workflow

The original circle/rectangle workflow remains available through the
`RUN_*LEGACY*` switches in `main_pipeline.py`.

Set:

```python
training_dataset_mode = "legacy"
```

Do not run legacy and composed actions in the same invocation. The pipeline
checks this and stops before writing data.
