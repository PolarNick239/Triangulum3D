#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import time


class DoubleClickHandler:

    def __init__(self, threshold=0.2):
        self._last_click = {}
        self._threshold = threshold
        self._cb = None

    def on_pressed(self, key):
        cur_time = time.time()
        double_click = False
        if key in self._last_click:
            delta = cur_time - self._last_click[key]
            if delta < self._threshold:
                double_click = True
        self._last_click[key] = cur_time
        if double_click and self._cb is not None:
            self._cb(key)
        return double_click

    def set_double_click_callback(self, cb):
        self._cb = cb
