#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import numpy as np
from pathlib import Path

from triangulum.utils import shapes
from triangulum.utils.shapes import draw_pixels
from triangulum_test.test_support import TestBase
from triangulum.algos.canny.edge_detection import EdgeDetectionProcessor
from triangulum.algos.ellipses_detection.ellipse_fitting import fit_ellipse

logger = logging.getLogger(__name__)


class FitEllipseTest(TestBase):

    def setUp(self):
        super().setUp()
        self.processor = EdgeDetectionProcessor(debug_enabled=True)

    def _by_points(self, case_name, n):
        w, h = 300, 150
        ellipse = shapes.Ellipse(w // 2, h // 2, w // 3, h // 3, 20)

        img_contour = np.zeros((h, w), np.uint8)
        ellipse.draw(img_contour, 255)
        self.dump_debug_img(Path(case_name) / '0_ellipse.png', img_contour)

        np.random.seed(239)
        angles = np.random.randint(0, 359, (n,))
        xys = np.float32(list(map(ellipse.calculate_point, angles)))
        img_points = img_contour // 3
        draw_pixels(img_points, xys, 255)
        self.dump_debug_img(Path(case_name) / '1_points.png', img_points)

        estimation = fit_ellipse(xys)
        estimation.draw(img_points, 255)
        self.dump_debug_img(Path(case_name) / '2_estimation.png', img_points)

        self.assertTrue(np.all(np.abs(np.array(ellipse) - np.array(estimation)) < 1.0))

    def five_points_test(self):
        self._by_points('5_points', 5)

    def ten_points_test(self):
        self._by_points('10_points', 10)
