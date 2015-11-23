#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.utils import support
from triangulum.third_party import transformations


def norm2(a):
    return (a * a).sum(-1)


def norm(a):
    return np.sqrt(norm2(a))


def normalize(a):
    return a / norm(a)


def homogenize(a, w=1.0):
    """
    Example:
        a=[
            [a00, a01],
            [a10, a11],
            [a20, a21]
        ], w=1
         ->
        result=[
            [a00, a01, w],
            [a10, a11, w],
            [a20, a21, w]
        ]
    """
    return np.hstack([a, np.full((len(a), 1), w, a.dtype)])


def homo_translate(matrix, points):
    points = np.atleast_2d(points)
    if points.shape[-1] < matrix.shape[1]:
        points = homogenize(points)
    p = np.dot(points, matrix.T)
    return p[:, :-1] / p[:, -1, np.newaxis]


def scale_matrix(s, d=2):
    if np.isscalar(s):
        s = np.array([s]*d)
    return np.diag(np.hstack([s, 1.0]))


def rotate_matrix2d(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha),  np.cos(alpha), 0],
                     [            0,              0, 1]])


# def apply_matrix_to(matrix, indicies, dim): TODO: implement
#     n, m = matrix.shape
#     assert n == m
#
#     indicies = list(indicies)
#     for i in range(n):
#         if i not in indicies:
#             indicies.append(i)
#
#     pre_permutation = np.zeros((n, n), np.int32)
#     for i, j in enumerate(indicies):
#         pre_permutation[i, j] = 1
#
#     return np.dot(np.linalg.inv(pre_permutation), np.dot(matrix, pre_permutation))


def look_at_matrix(eye, target, up=(0, 0, 1), right=None):
    """
    Camera frustum looks along -Z axis. See gluLookAt.
    """
    # TODO: review
    forward = np.float64(target) - eye
    forward = normalize(forward)
    if np.allclose(target[:2], eye[:2]) and up[2] == 1:
        if right is not None:
            right = normalize(right)
        else:
            right = normalize(np.array([1, 0, 0]))
    else:
        right = normalize(np.cross(forward, up))
    down = np.cross(forward, right)
    R = np.float64([right, -down, -forward])
    tvec = -np.dot(R, eye)
    return np.float32(np.vstack([np.column_stack([R, tvec]), [0, 0, 0, 1]]))


def ortho_matrix(aspect, near, far, width):
    """
    Camera frustum looks along -Z axis.
    Result frustum camera looks along -Z axis, like in OpenGL.
    """
    height = aspect * width
    P = transformations.clip_matrix(-width/2, width/2, -height/2, height/2, near, far, perspective=False)
    P = np.dot(P, scale_matrix([1, 1, -1]))
    return np.float32(P)


def perspective_matrix(aspect, near, far, fov_h=45):
    """
    Camera frustum looks along -Z axis.
    Result frustum camera looks along -Z axis, like in OpenGL.
    """
    tan = np.tan(np.radians(fov_h) / 2)
    right = tan * near
    left = -right
    bottom, top = aspect * left, aspect * right
    P = transformations.clip_matrix(left, right, bottom, top, near, far, perspective=True)
    P = np.dot(P, scale_matrix([1, 1, -1]))
    return np.float32(-P)
