#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import yaml
import logging
import pkg_resources
from pathlib import Path
from unittest import TestCase

from triangulum.utils.support import str_dict, deep_merge


logger = logging.getLogger(__name__)


resources_dir_path = Path(pkg_resources.get_provider('triangulum_test.resources').get_resource_filename(__name__, '.'))

_default_test_config = {
    "debug_output_dir": None,
}


def load_config():
    config_path = str(resources_dir_path / "test_config.yml")
    try:
        with open(config_path) as f:
            user_config = yaml.load(f)
            config = deep_merge(_default_test_config, user_config)
            logger.debuf("Using test config:\n{}".format(str_dict(config)))
    except FileNotFoundError:
        config = _default_test_config
        logger.debug("No config file found at '{}'.".format(config_path))
        logger.debug("Using test config (default one):\n{}".format(str_dict(config)))
    return config


class TestBase(TestCase):

    def setUp(self):
        super().setUp()
        self.config = load_config()

    def tearDown(self):
        super().tearDown()
