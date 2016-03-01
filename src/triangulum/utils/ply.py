#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
from pathlib import Path


def read_ply(path):
    path = Path(path)

    with path.open('rb') as file:
        prop_to_dtype = {'float': np.float32, 'int': np.int32, 'uchar': np.uint8}

        header = []
        while True:
            words = [word.decode() for word in file.readline().split()]
            if len(words) == 1 and words[0] == 'ply':
                continue
            header.append(words)
            if words[0] == 'end_header':
                break

        it = iter(header)
        words = it.__next__()
        elements = {}
        while True:
            if words[0] == 'end_header':
                break
            if words[0] == 'element':
                element_name, element_len = words[1], int(words[2])
                element_props = []
                words = it.__next__()
                while words[0] == 'property':
                    element_props.append(words)
                    words = it.__next__()
                if element_name == 'face':
                    el_type = np.dtype([('count', np.uint8), ('idx', np.int32, 3)])
                    elements[element_name] = np.fromfile(file, el_type, element_len)['idx'].copy()
                else:
                    el_type = np.dtype([(name, np.dtype(prop_to_dtype[tp])) for _, tp, name in element_props])
                    elements[element_name] = np.fromfile(file, el_type, element_len)
                continue
            words = it.__next__()

        v = elements['vertex']
        xyz, rgb, nxyz = ['x', 'y', 'z'], ['red', 'green', 'blue'], ['nx', 'ny', 'nz']
        if set(v.dtype.fields).issuperset(xyz):
            elements['xyz'] = v[xyz].view((np.float32, 3))
        if set(v.dtype.fields).issuperset(nxyz):
            elements['normal'] = v[nxyz].view((np.float32, 3))
        if set(v.dtype.fields).issuperset(rgb):
            elements['rgb'] = v[rgb].view((np.uint8, 3))

        # elements.keys():
        #   xyz - points coordinates
        #   face - face indices (triangles, so three vertex indices int32 value)
        return elements


def write_ply(path, points, faces=None):
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

        faces_data = None
        if faces is not None:
            f_print("element face {}".format(len(faces)))
            f_print("property list uchar int vertex_indices")

            face_t = np.dtype([('vn', np.uint8), ('vs', np.uint32, 3)])
            faces_data = np.zeros(len(faces), face_t)
            faces_data['vn'][:] = 3
            faces_data['vs'] = faces

        f_print('end_header')

        points.tofile(f)
        if faces_data is not None:
            faces_data.tofile(f)
