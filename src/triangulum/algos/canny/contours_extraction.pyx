#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

# distutils: language = c++

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef dfs(np.ndarray[np.uint8_t, ndim=2] state, int x, int y,
         vector[vector[pair[int, int]]]& contours,
         vector[pair[int, int]]& current_contour):
    cdef int h, w, dx, dy
    h, w = state.shape[0], state.shape[1]

    cdef int EMPTY = 0, INITED = 2, IN_CONTOUR = 3
    cdef int[8][2] dxys = [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, -1],
        [-1, 1],
    ];
    cdef int* dxy
    if state[y, x] != INITED:
        return

    current_contour.push_back(pair[int, int](x, y))
    state[y, x] = IN_CONTOUR
    for dxy in dxys:
        dx, dy = dxy[0], dxy[1]
        if not (0 <= y + dy < h and 0 <= x + dx < w):
            continue
        if state[y + dy, x + dx] == INITED:
            dfs(state, x + dx, y + dy, contours, current_contour)
            current_contour.clear()
    if current_contour.size() > 0:
        contours.push_back(current_contour)


def extract_contours(np.ndarray[np.uint8_t, ndim=2] is_edge):
    cdef int w, h, x, y, sy, sx, cx, cy, dx, dy, deg, minDeg, bestX, bestY, i
    cdef bool cornerFound
    is_edge = is_edge.copy()
    h, w = is_edge.shape[0], is_edge.shape[1]

    cdef int EMPTY = 0, IS_NEW = 1, INITED = 2
    cdef int[3] dxs = [-1, 0, 1]

    cdef vector[vector[pair[int, int]]] contours
    cdef vector[pair[int, int]] contour

    cdef vector[pair[int, int]] queue
    cdef vector[pair[int, int]] corners

    for sy in range(h):
        for sx in range(w):
            if is_edge[sy, sx] == IS_NEW:
                queue.clear()

                cornerFound = False
                minDeg = 9
                bestX, bestY = -1, -1

                queue.push_back(pair[int, int](sx, sy))
                is_edge[sy, sx] = INITED

                while len(queue) > 0:
                    nextXY = queue.back()
                    x, y = nextXY.first, nextXY.second
                    queue.pop_back()

                    deg = 0
                    for dy in dxs:
                        for dx in dxs:
                            if dx == 0 and dy == 0:
                                continue
                            if 0 <= y + dy < h and 0 <= x + dx < w:
                                if is_edge[y + dy, x + dx] == IS_NEW or is_edge[y + dy, x + dx] == INITED:
                                    deg += 1
                                if is_edge[y + dy, x + dx] == IS_NEW:
                                    queue.push_back(pair[int, int](x + dx, y + dy))
                                    is_edge[y + dy, x + dx] = INITED
                    if deg == 1:
                        corners.push_back(pair[int, int](x, y))
                        cornerFound = True
                    elif deg < minDeg:
                        minDeg = deg
                        bestX, bestY = x, y

                if not cornerFound:
                    corners.push_back(pair[int, int](bestX, bestY))

    cdef pair[int, int] corner
    cdef vector[pair[int, int]] current_contour
    for corner in corners:
        current_contour.clear()
        dfs(is_edge, corner.first, corner.second, contours, current_contour)

    np_contours = []
    cdef pair[int, int] xy
    cdef np.ndarray[np.int32_t, ndim=2] np_contour
    for contour in contours:
        np_contour = np.zeros((contour.size(), 2), np.int32)
        for i in range(contour.size()):
            np_contour[i, 0] = contour[i].first
            np_contour[i, 1] = contour[i].second
        np_contours.append(np_contour)
    return np_contours
