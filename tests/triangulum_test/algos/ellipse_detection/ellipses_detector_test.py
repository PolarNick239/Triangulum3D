#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import logging

import numpy as np

from triangulum.algos.ellipses_detection.ellipses_detector import EllipsesDetector
from triangulum.utils import shapes
from triangulum_test.test_support import TestBase

logger = logging.getLogger(__name__)


class EllipsesDetectorTest(TestBase):

    def simple_test(self):
        w, h = 500, 100

        edges = np.zeros((h, w), np.uint8)
        ellipse1 = shapes.Ellipse(w // 6, h // 2, w // 9, h // 3, 20)
        ellipse2 = shapes.Ellipse(5 * w // 6, h // 2, w // 9, h // 3, 360-40)
        ellipses = [ellipse1, ellipse2]

        for ellipse in ellipses:
            ellipse.draw(edges, 1, min_angle=0, max_angle=180+45)

        self.dump_debug_img('1_input_ellipses.png', edges * 255)

        processor = EllipsesDetector()

        ellipses_res = processor.process_edges(edges)

        img = np.zeros((h, w), np.uint8)
        for estimate in ellipses_res:
            estimate.draw(img, 255)
        self.dump_debug_img('2_estimates.png', img)

        self.assertEqual(len(ellipses_res), 2)

        is_found = [False] * len(ellipses)
        for estimate in ellipses_res:
            for i, ellipse in enumerate(ellipses_res):
                if np.all(np.abs(np.array(estimate) - np.array(ellipse)) < 1.0):
                    is_found[i] = True

        self.assertTrue(np.all(is_found))
