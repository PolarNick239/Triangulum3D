#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum_test.test_support import TestBase
from triangulum.scanner.central_line_extraction import CentralLineExtractionProcessor


class CentralLineExtractionProcessorTest(TestBase):

    def central_line_extraction_case(self, w, h):
        half_w = w / 2 - (1 - w % 2)

        processor = CentralLineExtractionProcessor()
        img = np.zeros((h, w), np.int32)
        img[:, :] = np.arange(h * w).reshape(h, w)
        img[:h / 2] = 1
        img[h / 2:] = 2
        is_center = processor.process(img)
        self.assertTrue(is_center.shape == img.shape)
        is_center2 = processor.process(img.T).T
        self.assertTrue(np.all(is_center == is_center2))

        self.assertTrue(np.all(is_center[:h / 2][half_w:-half_w, half_w:-half_w]))
        self.assertTrue(np.all(is_center[h / 2:][half_w:-half_w, half_w:-half_w]))

        self.assertFalse(np.all(is_center[:, :half_w]))
        self.assertFalse(np.all(is_center[:, -half_w + 1:]))

    def central_line_test_simple_odd(self):
        self.central_line_extraction_case(5, 100)

    def central_line_test_simple_even(self):
        self.central_line_extraction_case(6, 100)
