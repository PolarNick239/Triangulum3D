#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.algos.canny.contours_extraction import extract_contours
from triangulum.utils.math import homo_translate, rotate_matrix2d
from triangulum.algos.ellipses_detection.ellipse_fitting import fit_ellipse


class EllipsesDetector:

    def __init__(self):
        self._err_threshold = 0.05
        self._min_sectors_fraction = 0.6
        self._min_pixels = 40
        self._min_radius = 3
        self.ellipses = []

    def _is_ok(self, ellipse):
        if ellipse is None:
            return False
        center, axes, angle = (ellipse.x, ellipse.y), (ellipse.a, ellipse.b), ellipse.angle
        if max(axes) < self._min_radius or min(axes) == 0:
            return False
        if max(axes) / min(axes) > 10.0:
            return False
        return True

    def _transformation_ellipse_to_circle(self, ellipse):
        center, axes, angle = (ellipse.x, ellipse.y), (ellipse.a, ellipse.b), ellipse.angle

        rotate = rotate_matrix2d(np.deg2rad(-angle))
        move_to_zeros = [[1, 0, -center[0]],
                         [0, 1, -center[1]],
                         [0, 0, 1]]
        scale = [[1 / axes[0], 0, 0],
                 [0, 1 / axes[1], 0],
                 [0, 0, 1]]
        move_to_offset = [[1, 0, center[0]],
                          [0, 1, center[1]],
                          [0, 0, 1]]
        return np.dot(move_to_offset, np.dot(scale, np.dot(rotate, move_to_zeros)))

    def _distance_to_ellipse_as_circle(self, points, ellipse):
        to_circle = self._transformation_ellipse_to_circle(ellipse)

        points = homo_translate(to_circle, points)
        center, axes, angle = (ellipse.x, ellipse.y), (ellipse.a, ellipse.b), ellipse.angle
        center = homo_translate(to_circle, center)
        return np.abs(np.linalg.norm(points - center, axis=-1) - 1.0)

    def _calculate_sectors(self, points, ellipse, sectors_number):
        points = points.reshape(-1, 2)
        to_circle = self._transformation_ellipse_to_circle(ellipse)

        points = homo_translate(to_circle, points)
        center, axes, angle = (ellipse.x, ellipse.y), (ellipse.a, ellipse.b), ellipse.angle
        center = homo_translate(to_circle, center)
        points = points - center
        return np.unique((sectors_number * (np.arctan2(points[:, 1], points[:, 0]))) // (2 * np.pi))

    def process_contour(self, contour):
        points = contour.reshape(-1, 2)
        found_ellipses = []
        n = len(points)
        mask = np.ones(n, np.bool)

        if n < self._min_pixels:
            return found_ellipses

        for ellipse in self.ellipses:
            err = self._distance_to_ellipse_as_circle(points, ellipse)
            mask[err < self._err_threshold] = False

        for from_i in range(0, len(points), self._min_pixels // 4):
            number_to_sample = 3 * len(points) // 5
            sub_points = points[from_i:][:number_to_sample]
            sub_points = sub_points[mask[from_i:][:number_to_sample]]
            if len(sub_points) < self._min_pixels // 2:
                continue

            max_inliers = 0
            best_ellipse = None
            for i in range(20):
                indices = np.arange(len(sub_points))
                np.random.shuffle(indices)
                sample_points = sub_points[indices[:self._min_pixels // 2]]
                assert len(sample_points) >= 5

                ellipse = fit_ellipse(sample_points)
                if not self._is_ok(ellipse):
                    continue
                err = self._distance_to_ellipse_as_circle(points[mask], ellipse)
                inliers_mask = err < self._err_threshold
                inliers = inliers_mask.sum()

                if inliers < self._min_pixels:
                    continue

                if inliers > max_inliers:
                    max_inliers = inliers
                    best_ellipse = ellipse

            if best_ellipse is not None:
                err = self._distance_to_ellipse_as_circle(points[mask], best_ellipse)
                inliers_mask = err < self._err_threshold

                for i in range(3):
                    if inliers_mask.sum() < 5:
                        break
                    ellipse = fit_ellipse(points[mask][inliers_mask])
                    if not self._is_ok(ellipse):
                        break
                    err = self._distance_to_ellipse_as_circle(points[mask], ellipse)
                    inliers_mask = err < self._err_threshold

                if inliers_mask.sum() < 5 or not self._is_ok(ellipse):
                    break
                sectors_number = self._min_pixels // 2
                covered_sectors_number = len(self._calculate_sectors(points[mask][inliers_mask], ellipse, sectors_number))

                if covered_sectors_number / sectors_number < self._min_sectors_fraction:
                    continue

                err = self._distance_to_ellipse_as_circle(points, ellipse)
                mask[err < self._err_threshold] = False
                self.ellipses.append(ellipse)
                found_ellipses.append(ellipse)
        return found_ellipses

    def process_edges(self, is_edge):
        contours = extract_contours(is_edge)
        for contour in contours:
            self.process_contour(contour)
        return self.ellipses
