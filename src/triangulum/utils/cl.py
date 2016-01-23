#
# Copyright (c) 2015, Transas
# All rights reserved.
#

import logging
import pyopencl as cl

logger = logging.getLogger(__name__)


def create_context():
    platforms = cl.get_platforms()
    logger.debug('OpenCL platforms: {}'.format(['{}: {}.'.format(platform.get_info(cl.platform_info.VENDOR),
                                                                platform.get_info(cl.platform_info.NAME))
                                                for platform in platforms]))
    device_to_use = None
    for device_type, type_str in [(cl.device_type.GPU, 'GPU'), (cl.device_type.CPU, 'CPU')]:
        for platform in platforms:
            devices = platform.get_devices(device_type)
            if len(devices) > 0:
                logger.debug('OpenCL {} devices in {}: {}.'.format(type_str,
                                                                   platform.get_info(cl.platform_info.NAME),
                                                                   [device.get_info(cl.device_info.NAME) for device in devices]))
                if device_to_use is None:
                    device_to_use = devices[0]
                    logger.info('OpenCL device to use: {} {}'.format(platform.get_info(cl.platform_info.NAME),
                                                                     devices[0].get_info(cl.device_info.NAME)))
    if device_to_use is None:
        raise Exception('No OpenCL CPU or GPU device found!')
    return cl.Context([device_to_use])
