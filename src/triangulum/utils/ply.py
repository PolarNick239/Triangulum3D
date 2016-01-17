#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

from pathlib import Path


def write_to_ply(path, points):
    path = Path(path)
    points = points.reshape(-1, 3)

    dtype_to_type = {'float32': 'float',
                     'float64': 'double'}

    with path.open('wb') as f:
        def f_print(line):
            f.write((line + '\n').encode())

        f_print('ply')
        f_print('format binary_little_endian 1.0')

        f_print('element vertex {}'.format(len(points)))
        for property_name in ['x', 'y', 'z']:
            f_print('property {} {}'.format(dtype_to_type[points.dtype.name], property_name))

        f_print('end_header')

        points.tofile(f)
