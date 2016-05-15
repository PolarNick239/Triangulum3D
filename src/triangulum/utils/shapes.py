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

    def calculate_point(self, inner_angle):
        xy = self.a * np.cos(np.deg2rad(inner_angle)), self.b * np.sin(np.deg2rad(inner_angle))
        xy = math.homo_translate(math.rotate_matrix2d(np.deg2rad(self.angle)), xy)
        xy = np.array([self.x, self.y]) + xy
        return xy

    def draw(self, img, color, *,
             fill=False, min_angle=0, max_angle=360):
        angle = min_angle
        angle_step = 180 / 10
        prev_xy = None

        calc_xy = lambda inner_angle: np.int32(np.round(self.calculate_point(inner_angle)))

        def dist_from_prev(cur_xy):
            return np.abs(cur_xy - prev_xy).max()

        xys = []
        while angle < max_angle:
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
            xys.append(xy)

        if fill:
            xys = map(tuple, xys)
            xys = sorted(xys)

            cur_index = 0
            while cur_index < len(xys):
                cur_x, min_y = xys[cur_index]
                while cur_index + 1 < len(xys) and cur_x == xys[cur_index + 1][0]:
                    cur_index = cur_index + 1
                for y in range(min_y, xys[cur_index][1] + 1):
                    draw_pixels(img, (cur_x, y), color)
                cur_index = cur_index + 1
        return img
