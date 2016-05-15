#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
from pathlib import Path

from triangulum_test.test_support import TestBase
from triangulum.algos.central_line_extraction.central_line_extraction import CentralLineExtractionProcessor


class CentralLineExtractionProcessorTest(TestBase):

    def central_line_extraction_case(self, case_name, w, h):
        half_w = w / 2 - (1 - w % 2)

        processor = CentralLineExtractionProcessor()
        img = np.zeros((h, w), np.int32)
        img[:, :] = np.arange(h * w).reshape(h, w)
        img[:h / 2] = 1
        img[h / 2:] = 2

        self.dump_debug_img(Path(case_name) / '0_input.png', np.uint8(img * 127))
        is_center = processor.process(img)
        self.dump_debug_img(Path(case_name) / '1_is_center.png', np.uint8(is_center) * 255)

        self.assertTrue(is_center.shape == img.shape)
        is_center2 = processor.process(img.T).T
        self.assertTrue(np.all(is_center == is_center2))

        self.assertTrue(np.all(is_center[:h / 2][half_w:-half_w, half_w:-half_w]))
        self.assertTrue(np.all(is_center[h / 2:][half_w:-half_w, half_w:-half_w]))

        self.assertFalse(np.all(is_center[:, :half_w]))
        self.assertFalse(np.all(is_center[:, -half_w + 1:]))

    def central_line_test_simple_odd(self):
        self.central_line_extraction_case('odd', 5, 100)

    def central_line_test_simple_even(self):
        self.central_line_extraction_case('even', 6, 100)
