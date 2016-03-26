#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
import collections

from triangulum.utils import math


def draw_pixels(img, xy, color):
    h, w = img.shape[:2]
    xy = np.atleast_2d(xy)
    xy = np.int32(xy)
    mask = np.logical_and(np.logical_and(xy[:, 0] >= 0, xy[:, 0] < w),
                          np.logical_and(xy[:, 1] >= 0, xy[:, 1] < h))
    xy = xy[mask]
    img[xy.T[1], xy.T[0]] = color
    return img


class Ellipse(collections.namedtuple('Ellipse',
                                     ['x', 'y', 'a', 'b', 'angle'])):
    """
    (x, y) - ellipse center
    a, b - major and minor axis length
    angle - angle of ellipse in degrees
    """

    def draw(self, img, color):
        angle = 0
        angle_step = 180 / 10
        prev_xy = None

        def calc_xy(cur_angle):
            xy = self.a * np.cos(np.deg2rad(cur_angle)), self.b * np.sin(np.deg2rad(cur_angle))
            xy = math.homo_translate(math.rotate_matrix2d(np.deg2rad(self.angle)), xy)
            xy = np.array([self.x, self.y]) + xy
            return np.int32(xy)

        def dist_from_prev(cur_xy):
            return np.abs(cur_xy - prev_xy).max()

        while angle < 360:
            if prev_xy is None:
                xy = calc_xy(angle)
            else:
                xy = calc_xy(angle + angle_step)
                while dist_from_prev(xy) <= 1:
                    angle_step *= 2
                    xy = calc_xy(angle + angle_step)
                while dist_from_prev(xy) > 1:
                    angle_step /= 2
                    xy = calc_xy(angle + angle_step)

                angle += angle_step

            draw_pixels(img, xy, color)
            prev_xy = xy
