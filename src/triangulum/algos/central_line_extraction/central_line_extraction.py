#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
import pyopencl as cl
import pyopencl.array

from triangulum.utils import support
from triangulum.utils.cl import create_context


class CentralLineExtractionProcessor:
    """ This is OpenCL implementation for central line extraction from multiple stripes.
     The main idea is to calculate distance to stripe border,
     than apply non-maximum suppression on distance image (like in Canny edge detector)."""

    def __init__(self, cl_context: cl.Context = None, debug_enabled=False):
        self._context = cl_context

        self._kernel_source = None
        self._compiled_params = None

        self._debug_last_values = {'is_edge_pixel': None,
                                   'distance': None,
                                   'is_maximum': None} if debug_enabled else None

    def _compile(self, w, h):
        if self._compiled_params == (w, h):
            return

        self._kernel_source = self._kernel_source or support.load_kernel('triangulum.algos.central_line_extraction',
                                                                         'central_line_extraction')
        self._context = self._context or create_context()
        self._program = cl.Program(self._context, self._kernel_source).build(
                options=['-D W={}'.format(w), '-D H={}'.format(h)])
        self._compiled_params = (w, h)

    def process(self, class_img, no_class=np.iinfo(np.int32).min):
        assert class_img.dtype == np.int32
        assert not np.any(class_img == no_class)
        h, w = class_img.shape[:2]

        self._compile(w + 2, h + 2)

        bordered = np.full((h + 2, w + 2), no_class, np.int32)
        bordered[1:-1, 1:-1] = class_img

        queue = cl.CommandQueue(self._context)

        type_cl = cl.array.to_device(queue, bordered)
        is_edge_pixel_cl = cl.array.zeros_like(type_cl)
        distance_cl = cl.array.zeros(queue, bordered.shape, np.float32)
        distance_cl[:] = w + h + 239.0
        is_maximum_cl = cl.array.zeros_like(type_cl)

        self._program.detect_edge_pixels(queue, (w, h), None,
                                         type_cl.data, is_edge_pixel_cl.data,
                                         global_offset=(1, 1))

        changed = cl.array.zeros(queue, 1, np.int32)
        changed[0] = 1
        while np.any(changed.get()):
            changed[0] = 0
            self._program.nearest_edge_iter(queue, (w, h), None,
                                            type_cl.data, is_edge_pixel_cl.data, distance_cl.data, changed.data,
                                            global_offset=(1, 1))

        if self._debug_last_values is not None:
            self._debug_last_values['is_edge_pixel'] = is_edge_pixel_cl.get()[1:-1, 1:-1]

        distance_dxy_cl = cl.array.zeros(queue, distance_cl.shape, cl.array.vec.float2)
        self._program.convolve_sobel(queue, (w - 2, h - 2), None,
                                     distance_cl.data, distance_dxy_cl.data,
                                     global_offset=(2, 2))

        if self._debug_last_values is not None:
            self._debug_last_values['distance'] = distance_cl.get()[1:-1, 1:-1]

        self._program.non_maximum_suppression(queue, (w - 2, h - 2), None,
                                              distance_cl.data, distance_dxy_cl.data, is_maximum_cl.data,
                                              global_offset=(2, 2))
        is_maximum = np.array(is_maximum_cl.get(), np.bool)

        if self._debug_last_values is not None:
            self._debug_last_values['is_maximum'] = is_maximum[1:-1, 1:-1]
        return is_maximum[1:-1, 1:-1]
