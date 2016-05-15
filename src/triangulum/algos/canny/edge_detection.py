#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
import pkg_resources
import pyopencl as cl
import pyopencl.array
from pathlib import Path

from triangulum.utils import colors
from triangulum.utils import support
from triangulum.utils.cl import create_context
from triangulum.algos.canny.strong_edges_propagation import propagate_strong_edges


class EdgeDetectionProcessor:
    """ This is OpenCL implementation of Canny edge detection. """

    def __init__(self, cl_context: cl.Context = None, debug_enabled=False):
        self._context = cl_context

        self._kernel_source = None
        self._compiled_params = None

        self._debug_last_values = {'gaussed_grayscale': None,
                                   'intensity_dx': None,
                                   'intensity_dy': None,
                                   'intensity_dnorm': None,
                                   'is_extremum': None,
                                   'weak_edges': None,
                                   'strong_edges': None} if debug_enabled else None

    def _save_debug_value(self, key, f):
        if self._debug_last_values:
            assert key in self._debug_last_values
            self._debug_last_values[key] = f()

    def _compile(self, w, h):
        if self._compiled_params == (w, h):
            return

        self._kernel_source = self._kernel_source or support.load_kernel('triangulum.algos.canny', 'edge_detection')
        self._context = self._context or create_context()
        self._program = cl.Program(self._context, self._kernel_source).build(
                options=['-D W={}'.format(w), '-D H={}'.format(h)])
        self._compiled_params = (w, h)

    def process(self, img, strong_threshold=5.0, weak_threshold=2.5):
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            img = np.float32(img)
        else:
            img = colors.rgb_to_grayscale(img)
        img = np.pad(img, ((2, 2), (2, 2)), mode='edge')

        self._compile(w, h)

        queue = cl.CommandQueue(self._context)

        img_cl = cl.array.to_device(queue, img)
        gaussed_img_cl = cl.array.zeros(queue, (h + 2, w + 2), np.float32)

        self._program.convolve_gaussian(queue, (w, h), None,
                                        img_cl.data, gaussed_img_cl.data)

        intensity_dxy_cl = cl.array.zeros(queue, (h, w), cl.array.vec.float2)

        self._program.convolve_sobel(queue, (w, h), None,
                                     gaussed_img_cl.data, intensity_dxy_cl.data)

        is_extremum_cl = cl.array.zeros(queue, (h, w), np.int32)

        self._program.non_maximum_suppression(queue, (w, h), None,
                                              gaussed_img_cl.data, intensity_dxy_cl.data, is_extremum_cl.data)

        intensity_dxy = intensity_dxy_cl.get()
        intensity_dxy = np.dstack([intensity_dxy['x'], intensity_dxy['y']])
        intensity_dxy_norm = np.sum(intensity_dxy**2, axis=-1)**0.5

        is_extremum = np.array(is_extremum_cl.get(), np.bool)

        edge_strength = np.zeros((h, w), np.int32)
        edge_strength[np.logical_and(is_extremum, intensity_dxy_norm >= weak_threshold)] = 1
        edge_strength[np.logical_and(is_extremum, intensity_dxy_norm >= strong_threshold)] = 2

        is_edge = propagate_strong_edges(edge_strength) == 1

        self._save_debug_value('gaussed_grayscale', lambda: gaussed_img_cl.get())
        self._save_debug_value('intensity_dx', lambda: intensity_dxy_cl.get()['x'])
        self._save_debug_value('intensity_dy', lambda: intensity_dxy_cl.get()['y'])
        self._save_debug_value('intensity_dnorm', lambda: intensity_dxy_norm)
        self._save_debug_value('is_extremum', lambda: is_extremum)
        self._save_debug_value('weak_edges', lambda: edge_strength == 1)
        self._save_debug_value('strong_edges', lambda: edge_strength == 2)

        return is_edge
