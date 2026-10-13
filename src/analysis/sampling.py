"""Plot actual stored design parameters, coloured by frozen dataset role."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_sampling(cfg):
    destination = cfg.output_root / 'figures' / cfg.case_tag / "sampling"
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for family in cfg.generate_geometry_families:
        path = cfg.geometry_dataset_path(family, cfg.geometry_cases_per_family[family])
        with np.load(path, allow_pickle=True) as data:
            records = [json.loads(value) for value in data["metadata_json"]]
            roles = data["roles"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for role in ("train", "val", "test"):
            chosen = [record for record, label in zip(records, roles) if label == role]
            axes[0].scatter([r["cx"] * 1000 for r in chosen], [r["cy"] * 1000 for r in chosen], s=12, label=role)
            parameter = next(iter(records[0]["parameters"]))
            axes[1].scatter([r["parameters"][parameter] for r in chosen], [r["rotation"] for r in chosen], s=12, label=role)
        axes[0].set(xlabel="Centre x (mm)", ylabel="Centre y (mm)", title=family)
        axes[1].set(xlabel=parameter + " (SI units)", ylabel="Rotation (radians)")
        for axis in axes:
            axis.legend()
        fig.tight_layout()
        output = destination / f"{family}_sampling.png"
        fig.savefig(output, dpi=cfg.publish_dpi)
        if cfg.save_pdf_figures:
            fig.savefig(output.with_suffix('.pdf'))
        if cfg.show_figures:
            plt.show()
        plt.close(fig)
        paths.append(str(output))
    return paths
