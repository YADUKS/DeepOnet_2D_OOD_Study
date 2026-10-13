"""One registry of supported training families, including user extensions."""
from .custom_shapes import TRAINING_SHAPES

BUILTIN_TRAINING = {'circle': 3, 'ellipse': 5, 'rectangle': 5, 'triangle': 6,
                    'rhombus': 5, 'trapezoid': 7, 'regular_polygon': 5,
                    't_shape': 7, 'cross_shape': 7}
OOD_NAMES = ('star', 'aircraft', 'leaf', 'fin_like')
for name, spec in TRAINING_SHAPES.items():
    if name in BUILTIN_TRAINING or name in OOD_NAMES or not name.isidentifier() or name.lower() != name:
        raise ValueError(f'Invalid or duplicate custom geometry name: {name}')
    if not {'dimensions', 'map_parameters', 'boundary', 'seed_offset', 'version'} <= spec.keys():
        raise ValueError(f'Incomplete geometry registration: {name}')
    if spec['dimensions'] < 3 or not callable(spec['map_parameters']) or not callable(spec['boundary']):
        raise ValueError(f'Invalid geometry callbacks/dimensions: {name}')
TRAINING_NAMES = tuple(BUILTIN_TRAINING) + tuple(TRAINING_SHAPES)

def custom_shape(name):
    return TRAINING_SHAPES.get(name)

def dimensions(name):
    return TRAINING_SHAPES[name]['dimensions'] if name in TRAINING_SHAPES else BUILTIN_TRAINING[name]
