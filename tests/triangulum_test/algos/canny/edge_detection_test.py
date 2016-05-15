#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import numpy as np
from pathlib import Path

from triangulum.utils import shapes
import triangulum.utils.support as support
from triangulum_test.test_support import TestBase, resources_dir_path
from triangulum.algos.canny.edge_detection import EdgeDetectionProcessor

logger = logging.getLogger(__name__)


class EdgeDetectionProcessorTest(TestBase):

    def setUp(self):
        super().setUp()
        self.processor = EdgeDetectionProcessor(debug_enabled=True)

    def _process(self, case_name, img):
        self.dump_debug_img(Path(case_name) / '1_input.png', img)

        edges = self.processor.process(img)

        self.dump_debug_img(Path(case_name) / '2_gaussed_grayscale.png', np.uint8(self.processor._debug_last_values['gaussed_grayscale']))
        self.dump_debug_img(Path(case_name) / '3_intencity_dx.png', np.uint8(127 + self.processor._debug_last_values['intensity_dx']))
        self.dump_debug_img(Path(case_name) / '4_intencity_dy.png', np.uint8(127 + self.processor._debug_last_values['intensity_dy']))
        self.dump_debug_img(Path(case_name) / '5_intencity_dnorm.png', np.uint8(self.processor._debug_last_values['intensity_dnorm']))
        self.dump_debug_img(Path(case_name) / '6_is_extremum.png', np.uint8(self.processor._debug_last_values['is_extremum']) * 255)
        self.dump_debug_img(Path(case_name) / '7_weak_edges.png', np.uint8(self.processor._debug_last_values['weak_edges']) * 255)
        self.dump_debug_img(Path(case_name) / '8_strong_edges.png', np.uint8(self.processor._debug_last_values['strong_edges']) * 255)

        self.dump_debug_img(Path(case_name) / '9_edges.png', edges * 255)
        return edges

    def lena_test(self):
        img = np.array(support.load_image(resources_dir_path / 'data' / 'lena.png'))
        edges = self._process('lena', img)

    def ellipse_test(self):
        w, h = 300, 150
        ellipse = shapes.Ellipse(w // 2, h // 2, w // 3, h // 3, 20)

        img_contour = np.zeros((h, w), np.uint8)
        ellipse.draw(img_contour, 255)
        self.dump_debug_img(Path('ellipse') / '0_ellipse.png', img_contour)

        img = np.zeros((h, w), np.uint8)
        ellipse.draw(img, 255, fill=True)

        edges = self._process('ellipse', img)

        edges_points = np.nonzero(edges)
        distances = []
        for y, x in zip(edges_points[0], edges_points[1]):
            distances.append(y)

            is_ok = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if y > 0 and y < h - 1 and x > 0 and x < w - 1 and img_contour[y + dy, x + dx]:
                        is_ok = True

            self.assertTrue(is_ok)

        self.assertGreater(len(edges_points[0]), len(img_contour.nonzero()[0]) * 0.70)
        self.assertLess(len(edges_points[0]), len(img_contour.nonzero()[0]) * 1.3)
