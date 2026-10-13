"""Optional training families. See docs/manual/adding_geometry.md.

Each entry supplies dimensions, map_parameters, boundary, seed_offset, version.
No FE or torch imports here; callbacks can import geometry helpers lazily.
"""
TRAINING_SHAPES = {}
