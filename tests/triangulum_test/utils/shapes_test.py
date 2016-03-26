#
# Copyright (c) 2016, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.utils import shapes

from triangulum_test.test_support import TestBase


class ShapesTest(TestBase):

    def draw_pixels_test_grayscale_one_point(self):
        w, h = 10, 6
        img = np.zeros((h, w), np.uint8)

        img2 = shapes.draw_pixels(img, [0, 0], 239)
        self.assertIs(img2, img)
        self.assertEqual(img[0, 0], 239)

        shapes.draw_pixels(img, [w - 1, 0], 240)
        self.assertEqual(img[0, w - 1], 240)

        for x, y in [[0, -1], [-1, 0], [0, h + 1], [w + 1, 0]]:
            img_copy = img.copy()
            shapes.draw_pixels(img_copy, [x, y], 30)
            self.assertTrue(np.all(img == img_copy))

    def draw_pixels_test_color_one_point(self):
        w, h = 10, 6
        img = np.zeros((h, w, 3), np.uint8)

        img2 = shapes.draw_pixels(img, [0, 0], (239, 0, 30))
        self.assertIs(img2, img)
        self.assertTrue(np.all(img[0, 0] == [239, 0, 30]))

        shapes.draw_pixels(img, [w - 1, 0], (0, 200, 15))
        self.assertTrue(np.all(img[0, w - 1] == [0, 200, 15]))

        for x, y in [[0, -1], [-1, 0], [0, h + 1], [w + 1, 0]]:
            img_copy = img.copy()
            shapes.draw_pixels(img_copy, [x, y], (124, 2, 6))
            self.assertTrue(np.all(img == img_copy))

    def draw_pixels_test_grayscale_multi_points(self):
        w, h = 10, 6
        img = np.zeros((h, w), np.uint8)

        img2 = shapes.draw_pixels(img, [[0, 0], [3, 0]], 1)
        self.assertIs(img2, img)
        self.assertEqual(img[0, 0], 1)
        self.assertEqual(img[0, 3], 1)
        self.assertEqual(img.sum(), 2)

        center_yx = [h//2, w//2]
        expected_img = shapes.draw_pixels(img.copy(), center_yx, 30)
        for x, y in [[0, -1], [-1, 0], [0, h + 1], [w + 1, 0]]:
            shapes.draw_pixels(img, [[x, y], center_yx], 30)
            self.assertTrue(np.all(img == expected_img))

    def draw_ellipse_test(self):
        w, h = 500, 250
        img = np.zeros((h, w), np.uint8)
        shapes.Ellipse(w // 4, h // 2, w // 2, w // 6, 20).draw(img, 255)
        self.dump_debug_img('ellipse.png', img)
