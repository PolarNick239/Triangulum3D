#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio
import logging

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncExecutor:

    def __init__(self, max_workers, loop: asyncio.events.AbstractEventLoop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers)

    def __del__(self):
        self.shutdown()

    @asyncio.coroutine
    def map(self, fn, *args):
        result = yield from self.loop.run_in_executor(self.executor, fn, *args)
        return result

    def shutdown(self, wait=True):
        if self.executor is None:
            return False
        else:
            self.executor.shutdown(wait=wait)
            self.executor = None
            return True


def make_exc_info(exception):
    if exception is None:
        return None
    assert isinstance(exception, BaseException)
    return type(exception), exception, exception.__traceback__


def wrap_exc(coro_or_future, logger_for_error=None, future_description: str=None):
    logger_for_error = logger_for_error or logger
    def check_for_exceptions(done_future: asyncio.Future):
        if not done_future.cancelled() and done_future.exception():
            logger_for_error.error('Future {}done with exception!'.format('' if not future_description else
                                                                          '({}) '.format(future_description)),
                                   exc_info=make_exc_info(done_future.exception()))

    future = asyncio.async(coro_or_future)
    future.add_done_callback(check_for_exceptions)
    return future
