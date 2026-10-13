# Figures

Keep selected documentation figures here. Generated sampling plots, evaluation
figures, models and metrics belong under `outputs/` and are ignored by Git.

`banner.png` is the GitHub README illustration; `banner.svg` is the vector copy.
Both are created with Matplotlib from stored geometry and FE data. Temperature
is interpolated from stored FE vertex values, not predicted by a trained model.
The SDF is recomputed from the stored boundary on a finer display grid. Family
silhouettes use independent scales.

The committed illustration uses the nine-family/four-OOD miniature verification
dataset. Its FE source fingerprint and display choices are in `banner.json`.
The underlying generated datasets remain Git ignored.

To regenerate after generating the configured training and OOD datasets, run
from the repository root:

```text
python docs/figures/make_banner.py --config study.json
```

The script reads existing cases only; it does not solve FE cases or train models.
For the original local verification data, use
`--config verification/workflow_extensions/all_shapes.json` instead.
