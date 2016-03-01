#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import yaml
import asyncio
import logging
import numpy as np
import pkg_resources
from pathlib import Path
from unittest import TestCase

from triangulum.utils import support
from triangulum.utils.support import str_dict, deep_merge
from triangulum.rendering.gl import RenderingAsyncExecutor


logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.DEBUG,
                    format='%(relativeCreated)d [%(threadName)s]\t%(name)s [%(levelname)s]:\t %(message)s')


resources_dir_path = Path(pkg_resources.get_provider('triangulum_test.resources').get_resource_filename(__name__, '.'))

_default_test_config = {
    'debug_output_dir': None,
}


def load_config():
    config_path = str(resources_dir_path / "test_config.yml")
    try:
        with open(config_path) as f:
            user_config = yaml.load(f)
            config = deep_merge(_default_test_config, user_config)
            logger.debug("Using test config:\n{}".format(str_dict(config)))
    except FileNotFoundError:
        config = _default_test_config
        logger.debug("No config file found at '{}'.".format(config_path))
        logger.debug("Using test config (default one):\n{}".format(str_dict(config)))
    return config


class TestBase(TestCase):

    def setUp(self):
        super().setUp()
        self.config = load_config()

        self.gl_executor = None
        self.releasables = []

    def get_gl_executor(self):
        if self.gl_executor is None:
            self.gl_executor = RenderingAsyncExecutor()
        return self.gl_executor

    def gl_executor_map(self, foo, *args):
        gl_executor = self.get_gl_executor()
        result = asyncio.get_event_loop().run_until_complete(gl_executor.map(foo, *args))
        return result

    def register_releasable(self, releasable):
        self.releasables.append(releasable)

    def with_debug_output(self):
        return self.config['debug_output_dir'] is not None

    def debug_dir(self):
        return Path(self.config['debug_output_dir']) / self.__class__.__name__

    def dump_debug_img(self, path, img):
        if self.with_debug_output():
            path = self.debug_dir() / path
            support.silent_make_dir(path.parent)
            support.save_image(path, img)

    def dump_debug_matrix_by_hue(self, path, mat):
        if self.with_debug_output():
            path = self.debug_dir() / path
            support.silent_make_dir(path.parent)
            img = support.array_to_rgb_by_hue(mat)[:, :, ::-1]
            img = np.uint8(img)
            support.save_image(path, img)

    def tearDown(self):
        super().tearDown()
        for releasable in self.releasables:
            self.gl_executor_map(releasable.release)
