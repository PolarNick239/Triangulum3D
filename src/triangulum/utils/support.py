#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio
import logging
import numpy as np
import pkg_resources
from pathlib import Path
from itertools import chain
from PIL.Image import open as image_open
from PIL.Image import fromarray as image_fromarray
from concurrent.futures import ThreadPoolExecutor

from triangulum.utils.colorsys_np import hsv_to_rgb

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


def str_dict(d: dict):
    """
    >>> str_dict({1: 1, "1": "1"})
    "{'1': '1', 1: 1}"
    >>> str_dict({(2, 3): "1", 1: "2", 2: "3", "a": "4", "b": {"1": 1}})
    "{'a': '4', 'b': {'1': 1}, (2, 3): '1', 1: '2', 2: '3'}"
    """
    def to_str(x):
        if isinstance(x, str):
            return "'" + x + "'"
        else:
            return str(x)

    result = "{"
    for i, key in enumerate(sorted(d.keys(), key=to_str)):
        value = d[key]

        if isinstance(value, dict):
            value = str_dict(value)
        else:
            value = to_str(value)

        if i > 0:
            result += ", "
        result += "{}: {}".format(to_str(key), value)
    return result + "}"


def deep_merge(base_dict: dict, new_dict: dict):
    """
    >>> str_dict(deep_merge({1: "old", 2: "only_old"}, {1: "new", 3: "only_new"}))
    "{1: 'new', 2: 'only_old', 3: 'only_new'}"
    >>> str_dict(deep_merge({"inner": {1: "old", 2: "only_old"}}, {"inner": {1: "new", 3: "only_new"}}))
    "{'inner': {1: 'new', 2: 'only_old', 3: 'only_new'}}"
    >>> result = deep_merge({1: 0, 2: 0, 3: 239, 'd': {"d1": 1, "d2": 2}}, {1: 1, 2: -1, 239: 2012, 'd': {"d1": 3, "d3": 4}})
    >>> str_dict(result)
    "{'d': {'d1': 3, 'd2': 2, 'd3': 4}, 1: 1, 2: -1, 239: 2012, 3: 239}"
    """
    result = {}
    for key in set(chain(base_dict.keys(), new_dict.keys())):
        if key in new_dict:
            value = new_dict[key]
        else:
            value = base_dict[key]
        if isinstance(value, dict) and key in base_dict and key in new_dict:
            value = deep_merge(base_dict[key], new_dict[key])
        result[key] = value
    return result


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


def silent_make_dir(path: Path):
    try:
        path.mkdir(parents=True)  # TODO: migrate to python3.5 (it has exist_ok param)
    except FileExistsError:
        pass


def array_to_grayscale(img):
    assert len(img.shape) == 2
    values_range = (img.max() - img.min())
    if values_range == 0:
        values_range = 1
    return np.uint8(255 * (img - img.min()) / values_range)


def array_to_rgb_by_hue(img):
    """
    >>> array_to_rgb_by_hue(np.array([[10, 15, 20, 10]]))
    array([[[ 150.,   75.,   75.],
            [  75.,  150.,   75.],
            [  75.,   75.,  150.],
            [ 150.,   75.,   75.]]])
    """
    assert len(img.shape) == 2

    delta = img.max() - img.min()
    if delta > 0:
        hue = (img - img.min()) / delta
    else:
        hue = img - img.min()
    hue *= 2 / 3

    saturation = np.full(hue.shape, 0.5, hue.dtype)
    value = np.full(hue.shape, 150, hue.dtype)
    hsv = np.dstack([hue, saturation, value])
    rgb = hsv_to_rgb(hsv)
    return rgb


def load_image(path):
    return image_open(str(path))


def save_image(path, img):
    image = image_fromarray(img)
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            image = image.convert('RGB')
        if img.shape[2] == 4:
            image = image.convert('RGBA')
    image.save(str(path))


def load_kernel(package, kernel_name):
    kernel_path = Path(pkg_resources.get_provider(package)
                       .get_resource_filename(__name__, '{}.cl'.format(kernel_name)))
    with kernel_path.open() as f:
        kernel_source = ''.join(f.readlines())
    return kernel_source
