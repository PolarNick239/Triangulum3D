#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import numpy as np
from pathlib import Path

from triangulum.utils import shapes
import triangulum.utils.support as support
from triangulum.utils.shapes import draw_pixels
from triangulum_test.test_support import TestBase, resources_dir_path
from triangulum.algos.canny.edge_detection import EdgeDetectionProcessor
from triangulum.algos.canny.contours_extraction import extract_contours

logger = logging.getLogger(__name__)


class ContoursExtractionTest(TestBase):

    def setUp(self):
        super().setUp()
        self.processor = EdgeDetectionProcessor(debug_enabled=True)

    def _process(self, case_name, img):
        self.dump_debug_img(Path(case_name) / '01_input.png', img)

        edges = self.processor.process(img)

        self.dump_debug_img(Path(case_name) / '02_gaussed_grayscale.png', np.uint8(self.processor._debug_last_values['gaussed_grayscale']))
        self.dump_debug_img(Path(case_name) / '03_intencity_dx.png', np.uint8(127 + self.processor._debug_last_values['intensity_dx']))
        self.dump_debug_img(Path(case_name) / '04_intencity_dy.png', np.uint8(127 + self.processor._debug_last_values['intensity_dy']))
        self.dump_debug_img(Path(case_name) / '05_intencity_dnorm.png', np.uint8(self.processor._debug_last_values['intensity_dnorm']))
        self.dump_debug_img(Path(case_name) / '06_is_extremum.png', np.uint8(self.processor._debug_last_values['is_extremum']) * 255)
        self.dump_debug_img(Path(case_name) / '07_weak_edges.png', np.uint8(self.processor._debug_last_values['weak_edges']) * 255)
        self.dump_debug_img(Path(case_name) / '08_strong_edges.png', np.uint8(self.processor._debug_last_values['strong_edges']) * 255)

        self.dump_debug_img(Path(case_name) / '09_edges.png', edges * 255)
        contours = extract_contours(edges)

        pointsNumber = 0

        h, w = img.shape[:2]
        extracted_edges = np.zeros((h, w, 3), np.uint8)
        np.random.seed(239)
        for contour in contours:
            colour = np.random.randint(60, 255, (3,))
            draw_pixels(extracted_edges, contour, colour)
            pointsNumber += len(contour)
            self.assertTrue(np.all(edges == draw_pixels(edges.copy(), contour, 1)))  # assertion, that all pixels from contours are from edges
        self.dump_debug_img(Path(case_name) / '10_extracted_edges.png', extracted_edges)

        self.assertEqual(pointsNumber, len(edges.nonzero()[0]))

        return contours

    def lena_test(self):
        img = np.array(support.load_image(resources_dir_path / 'data' / 'lena.png'))
        edges = self._process('lena', img)

    def ellipse_test(self):
        w, h = 300, 150
        ellipse = shapes.Ellipse(w // 2, h // 2, w // 3, h // 3, 20)

        img_contour = np.zeros((h, w), np.uint8)
        ellipse.draw(img_contour, 255)
        self.dump_debug_img(Path('ellipse') / '00_ellipse.png', img_contour)

        img = np.zeros((h, w), np.uint8)
        ellipse.draw(img, 255, fill=True)

        edges = self._process('ellipse', img)
