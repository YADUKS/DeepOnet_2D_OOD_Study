# Adding a training geometry

Training, composition, tuning and reporting discover families from
`src/fem/registry.py`. Add an entry in `src/fem/custom_shapes.py`; the shared
pipeline does not need another shape-specific branch.

For example, the following complete extension defines a kite-shaped cavity.
Place it in `custom_shapes.py` alongside any existing entries:

```python
import numpy as np

def kite_parameters(cfg, u, candidate_id):
    ranges = cfg.custom_geometry_ranges.get("kite", {})
    def sample(key, default, index):
        low, high = ranges.get(key, default)
        return low + float(u[index]) * (high-low)
    parameters = {
        "width": sample("width", [0.008, 0.012], 0),
        "height": sample("height", [0.006, 0.010], 1),
        "shoulder_fraction": sample("shoulder_fraction", [0.30, 0.60], 2),
    }
    rotation = cfg.geometry_rotation_min + u[3] * (
        cfg.geometry_rotation_max-cfg.geometry_rotation_min)
    return parameters, float(rotation), float(u[4]), float(u[5])

def kite_boundary(parameters, curve_segments):
    width, height = parameters["width"], parameters["height"]
    fraction = parameters["shoulder_fraction"]
    if width <= 0 or height <= 0 or not 0 < fraction < 1:
        raise ValueError("Invalid kite dimensions")
    shoulder_y = (fraction-.5)*height
    return np.array([[0, -.5*height], [.5*width, shoulder_y],
                     [0, .5*height], [-.5*width, shoulder_y]])

TRAINING_SHAPES = {
    "kite": {
        "dimensions": 6,
        "map_parameters": kite_parameters,
        "boundary": kite_boundary,
        "seed_offset": 200000,
        "version": "1",
    },
}
```

The mapper receives one Halton/LHS unit-cube row and returns parameters,
rotation (radians), and two unit coordinates for placement. Return positive
physical lengths in metres. The shared placement code maps the centre into
the feasible domain while respecting clearance. The boundary callback returns
ordered canonical vertices around the origin. Shared geometry code applies
rotation/translation, validates the polygon and computes the mesh and SDF.

Use a unique stable `seed_offset`; do not renumber existing families. Increase
the version if helper code affecting geometry changes. Direct mapper/boundary
source code and this family's configured ranges are fingerprinted automatically.
Import FE/ML libraries only inside the stages that require them, never here.

In `study.json`, add `kite` to generation/evaluation lists and add its counts to
`geometry_cases_per_family`, `training_geometry_source_sizes`,
`training_geometry_counts`, `validation_cases_per_family`, and
`test_cases_per_family`. The three training pool counts should match for the
standard study. Optional ranges can be set as:

```json
"custom_geometry_ranges": {
  "kite": {"width": [0.008, 0.012], "height": [0.006, 0.010]}
}
```

First generate a small isolated dataset, preview the shape and run `mesh-check`.
For example (without using your production config):

```text
python run.py generate --families kite --cases 4 --val-cases 2 --test-cases 2 --data-root verification/kite/data --results-root verification/kite/outputs
python run.py preview --families kite --cases 4 --val-cases 2 --test-cases 2 --family kite --case 0 --data-root verification/kite/data --results-root verification/kite/outputs
```

Existing built-in family source identities do not change when registering a new
family. Selecting it creates a new composition and new model/result folders;
old models keep their original dataset and geometry membership.
