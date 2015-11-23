#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import time
import asyncio


class FPSLimiter:

    def __init__(self, fps_limit=60):
        if fps_limit == 0:
            fps_limit = -60
        self._fps_limit = fps_limit
        self._last_frame_time = None
        self._frame_changed = asyncio.Future()
        self._frame_changed.set_result(True)

    @asyncio.coroutine
    def _call_cb_regullary(self, cb, calls_per_second):
        while True:
            cb()
            yield from asyncio.sleep(1.0 / calls_per_second)

    @asyncio.coroutine
    def ensure_frame_limit(self, poll_events_cb=None):
        poll_events_cb = poll_events_cb or (lambda: 239)
        if self._fps_limit > 0:
            if self._last_frame_time is None:
                frame_delay = 0
            else:
                cur_time = time.time()
                frame_delay = 1.0 / self._fps_limit - (cur_time - self._last_frame_time)
            yield from asyncio.sleep(frame_delay)
            self._last_frame_time = time.time()
            poll_events_cb()
        else:
            polls_per_second = -self._fps_limit

            if self._frame_changed is not None and not self._frame_changed.done():
                polling = asyncio.async(self._call_cb_regullary(poll_events_cb, polls_per_second))
                yield from self._frame_changed
                polling.cancel()
            poll_events_cb()
            self._frame_changed = asyncio.Future()

    def update(self):
        if not self._frame_changed.done():
            self._frame_changed.set_result(True)

    def stop(self):
        self._frame_changed.cancel()
