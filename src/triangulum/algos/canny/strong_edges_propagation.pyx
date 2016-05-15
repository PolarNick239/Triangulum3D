#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

# distutils: language = c++

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.pair cimport pair

def propagate_strong_edges(np.ndarray[np.int32_t, ndim=2] edge_strength, weak_edge=1, strong_edge=2):
    cdef int w, h, sx, sy, x, y, dx, dy
    cdef vector[pair[int, int]] queue
    cdef pair[int, int] nextXY
    cdef int weak_edge_ = weak_edge
    cdef int strong_edge_ = strong_edge
    cdef int[3] dxs = [-1, 0, 1]

    edge_strength = edge_strength.copy()
    h, w = edge_strength.shape[0], edge_strength.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] result = np.zeros((h, w), np.uint8)
    for sy in range(h):
        for sx in range(w):
            pass
            if edge_strength[sy, sx] == strong_edge_:

                queue.clear()

                queue.push_back(pair[int, int](sx, sy))
                edge_strength[sy, sx] = 0
                result[sy, sx] = 1

                while len(queue) > 0:
                    nextXY = queue.back()
                    x, y = nextXY.first, nextXY.second
                    queue.pop_back()

                    for dy in dxs:
                        for dx in dxs:
                            if 0 <= y + dy < h and 0 <= x + dx < w \
                                    and (edge_strength[y + dy, x + dx] == strong_edge_ or edge_strength[y + dy, x + dx] == weak_edge_):
                                queue.push_back(pair[int, int](x + dx, y + dy))
                                edge_strength[y + dy, x + dx] = 0
                                result[y + dy, x + dx] = 1

    return result
