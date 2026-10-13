"""Build the README banner from stored FE cases; no solving or training.

From the repository root, after generating all default families:
    python docs/figures/make_banner.py --config study.json
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import Settings
from src.analysis.preview import load_stored_preview_case
from src.data.design import file_digest
from src.fem.geometry_spec import signed_distance_to_polygon

INK, MUTED, BLUE, ORANGE = '#12263A', '#536779', '#087E8B', '#B76A18'
NAMES = {'regular_polygon': 'Regular\npolygon', 't_shape': 'T-shape',
         'cross_shape': 'Cross', 'fin_like': 'Fin-like'}


def load_case(cfg, family, group='training'):
    counts = cfg.ood_cases_per_family if group == 'ood' else cfg.geometry_cases_per_family
    return load_stored_preview_case(replace(cfg, preview_dataset_group=group,
        preview_dataset_family=family, preview_dataset_size=counts[family],
        preview_dataset_case_number=0))


def build_banner(cfg, output):
    training = [load_case(cfg, family) for family in cfg.generate_geometry_families]
    external = [load_case(cfg, family, 'ood') for family in cfg.generate_ood_families]
    hero = next((case for case in training if case['family'] == 'cross_shape'), training[0])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(15.4, 7.1), facecolor='white')
    fig.text(.04, .932, 'From cavity geometry to temperature', fontsize=23, weight='bold')
    fig.text(.04, .881, 'DEEPONET  /  2D STEADY HEAT CONDUCTION  /  GEOMETRY GENERALISATION',
             fontsize=10, color=MUTED)
    boundary = hero['boundary_points'] * 1000
    width, height = hero['Lx'] * 1000, hero['Ly'] * 1000
    axes = [fig.add_axes([.04 + .244*i, .475, .208, .32]) for i in range(4)]
    titles = ('01  Cavity geometry', '02  Finite element mesh', '03  Signed-distance input', '04  Temperature target')
    subtitles = ('Heat in at top; cooling at cavity', 'Stored NGSolve reference case', 'Negative inside the cavity', 'Stored FE vertex values')
    for i, (ax, title, subtitle) in enumerate(zip(axes, titles, subtitles)):
        ax.set_aspect('equal')
        ax.set(xlim=(0, width), ylim=(0, height))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#B6C6D0')
        fig.text(.04+.244*i, .824, title, fontsize=12, weight='bold')
        fig.text(.04+.244*i, .794, subtitle, fontsize=8.5, color=MUTED)

    axes[0].add_patch(Rectangle((0,0), width, height, facecolor='#E3F1F2', edgecolor='none'))
    axes[0].add_patch(Polygon(boundary, facecolor='white', edgecolor=BLUE, linewidth=2))
    for x in np.linspace(3, width-3, 7):
        axes[0].annotate('', (x, height-2.2), (x, height),
                         arrowprops={'arrowstyle': '-|>', 'color': '#CB6145', 'lw': 1.4})
    axes[0].text(width/2, 1.25, f'{width:.0f} × {height:.0f} mm plate', ha='center', color=MUTED, fontsize=9)
    points = hero['fe_points'] * 1000
    triangulation = mtri.Triangulation(points[:,0], points[:,1], hero['fe_triangles'])
    axes[1].triplot(triangulation, color='#6B8391', linewidth=.38)
    axes[1].add_patch(Polygon(boundary, fill=False, edgecolor=BLUE, linewidth=1.4))
    x, y = np.meshgrid(np.linspace(0, hero['Lx'], 240), np.linspace(0, hero['Ly'], 140))
    distance = signed_distance_to_polygon(x, y, hero['boundary_points']) * 1000
    sdf = axes[2].imshow(distance, origin='lower', extent=(0,width,0,height), cmap='RdBu',
                         vmin=-np.max(abs(distance)), vmax=np.max(abs(distance)))
    axes[2].contour(x*1000, y*1000, distance, levels=[0], colors=[INK], linewidths=.8)
    temperature = axes[3].tripcolor(triangulation, hero['fe_temperature'], shading='gouraud', cmap='inferno')
    axes[3].add_patch(Polygon(boundary, facecolor='white', edgecolor='white', linewidth=.7))
    for index, scalar, unit in ((2, sdf, 'Signed distance (mm)'), (3, temperature, 'Temperature (K)')):
        cax = fig.add_axes([.04 + .244*index, .434, .208, .015])
        bar = fig.colorbar(scalar, cax=cax, orientation='horizontal')
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, length=2, colors=MUTED)
        bar.set_label(unit, fontsize=8, color=MUTED, labelpad=2)

    fig.add_artist(plt.Line2D([.04,.98], [.348,.348], color='#D9E2E9', lw=1))
    groups = ((training, .04, .64, BLUE, 'Training / validation / test families'),
              (external, .724, .252, ORANGE, 'Held-out OOD families'))
    for cases, left, span, colour, heading in groups:
        fig.text(left, .300, heading, fontsize=11, weight='bold', color=colour)
        slot = span / max(len(cases), 1)
        for i, case in enumerate(cases):
            ax = fig.add_axes([left+i*slot+.003, .129, slot-.012, .125])
            vertices = case['boundary_points'].astype(float)
            centre = (vertices.max(axis=0) + vertices.min(axis=0))/2
            vertices = (vertices-centre)/np.ptp(vertices, axis=0).max()
            ax.add_patch(Polygon(vertices, facecolor=colour, edgecolor=colour, linewidth=.5))
            ax.set(xlim=(-.6,.6), ylim=(-.6,.6), aspect='equal')
            ax.axis('off')
            label = NAMES.get(case['family'], case['family'].replace('_',' ').capitalize())
            fig.text(left+(i+.5)*slot, .104, label, ha='center', va='top', fontsize=8.5, color=MUTED)
    fig.text(.04, .025, 'Illustration from stored verification data; shape silhouettes use independent scales. Temperature shown is an FE reference, not a trained-model prediction.',
             fontsize=8, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=170, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    provenance = {'purpose': 'Documentation illustration; no research accuracy claim.',
                  'hero_family': hero['family'], 'hero_metadata': hero['metadata'],
                  'hero_fe_sha256': file_digest(hero['fe_path']),
                  'sdf_display': 'Recomputed geometric signed distance on a 240 x 140 display grid, in mm.',
                  'temperature_display': 'Gouraud interpolation of stored FE mesh vertex temperatures in K.',
                  'training_families': [case['family'] for case in training],
                  'ood_families': [case['family'] for case in external]}
    output.with_suffix('.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Saved README banner: {output.with_suffix(".png")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('banner'))
    args = parser.parse_args()
    cfg = Settings(**json.loads(args.config.read_text(encoding='utf-8')))
    build_banner(cfg, args.output)
