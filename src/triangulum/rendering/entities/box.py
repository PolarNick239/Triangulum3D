#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

import triangulum.utils.math as math
from triangulum.rendering.entities.points_cloud import PointsCloud


class Box(PointsCloud):

    def __init__(self, xs, ys, zs, course=0, pitch=0, roll=0):
        center_x, center_y, center_z = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2
        xs, ys, zs = np.array(xs) - center_x, np.array(ys) - center_y, np.array(zs) - center_z

        verts = np.array([
            [xs[0], ys[0], zs[0]],
            [xs[1], ys[0], zs[0]],
            [xs[0], ys[1], zs[0]],
            [xs[1], ys[1], zs[0]],
            [xs[0], ys[0], zs[1]],
            [xs[1], ys[0], zs[1]],
            [xs[0], ys[1], zs[1]],
            [xs[1], ys[1], zs[1]]])

        rotate_course = math.rotate_matrix2d(np.radians([course])[0])
        rotate_pitch = math.rotate_matrix2d(np.radians([pitch])[0])
        rotate_roll = math.rotate_matrix2d(np.radians([roll])[0])

        verts[:, [1, 2]] = math.homo_translate(rotate_roll, verts[:, [1, 2]])
        verts[:, [0, 2]] = math.homo_translate(rotate_pitch, verts[:, [0, 2]])
        verts[:, [0, 1]] = math.homo_translate(rotate_course, verts[:, [0, 1]])
        verts += [center_x, center_y, center_z]

        faces = [[0, 1, 2], [1, 2, 3],
                 [4, 5, 6], [5, 6, 7],
                 [0, 1, 4], [1, 4, 5],
                 [2, 3, 6], [3, 6, 7],
                 [0, 2, 4], [2, 4, 6],
                 [1, 3, 5], [3, 5, 7]]

        super(Box, self).__init__(verts, faces=faces)
