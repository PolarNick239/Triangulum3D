#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np


def area(a):
    if a is None:
        return 0.0
    return np.prod(a[1]-a[0])


def intersects(a, b):
    return np.all(a[0] < b[1]) and np.all(b[0] < a[1])


def intersection(a, b):
    """
    >>> a = np.int32( [[0 , 0] , [200, 100]] )
    >>> intersection(a, a + 50)
    array([[ 50,  50],
           [200, 100]], dtype=int32)
    >>> intersection(a, a + 500) is None
    True
    """
    c = np.array([np.maximum(a[0], b[0]), np.minimum(a[1], b[1])])
    return c if np.all(c[0] < c[1]) else None


def aabb(points):
    points = np.asarray(points)
    dim = points.shape[-1]
    points = points.reshape(-1, dim)
    p1, p2 = np.nanmin(points, 0), np.nanmax(points, 0)
    return np.array([p1, p2])


def rect_to_quad(box):
    (x1, y1), (x2, y2) = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def contains_point(aabb, p):
    """
    >>> aabb = np.int32([[0 , 0] , [200, 100]])
    >>> p = np.int32([10, 50])
    >>> contains_point(aabb, p)
    True
    >>> contains_point(aabb, p + 1000)
    False
    """
    return np.all(aabb[0] <= p, -1) & np.all(p < aabb[1], -1)


def align(a, align_by):
    a = np.int64(a).copy()
    a[0] = a[0] // align_by * align_by
    a[1] = (a[1] + align_by - 1) // align_by * align_by
    return a
