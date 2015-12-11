#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import numpy as np

from triangulum.utils import math
from triangulum.rendering.entities.camera import Camera
from triangulum.rendering.entities.stripes_projector import StripesProjector

logger = logging.getLogger(__name__)


class ReconstructionBuilder:

    def __init__(self):
        self._next_projector_lod = 0

        self._wh = None
        self._stripe_ids_img = None
        self._mask = None

    def _init(self, w, h):
        assert self._stripe_ids_img is None and self._mask is None and self._wh is None
        self._wh = (w, h)
        self._stripe_ids_img = np.zeros((h, w), np.int32)
        self._mask = np.ones((h, w), np.bool)

    def process_observation(self, projector_lod, img):
        assert self._next_projector_lod == projector_lod
        self._next_projector_lod += 1

        h, w = img.shape[:2]
        if projector_lod == 0:
            self._init(w, h)
        else:
            assert (w, h) == self._wh

        is_green, is_red, bad_mask = self._classify(img)
        self._mask[bad_mask] = 0
        self._stripe_ids_img = self._stripe_ids_img * 2 + is_green
        logger.debug('After projector lod={}: {}% points recognized.'
                     .format(projector_lod, 100 * self._mask.sum() // np.prod(self._mask.shape)))

    @staticmethod
    def _classify(img):
        h, w = img.shape[:2]
        is_green = img[:, :, 1] > 128
        is_red = img[:, :, 2] > 128
        is_green, is_red = np.logical_and(is_green, ~is_red), np.logical_and(is_red, ~is_green)
        bad = np.logical_and(~is_green, ~is_red)
        return is_green, is_red, bad

    def build_point_cloud(self, projector: StripesProjector, camera: Camera):
        w, h = self._wh

        # noinspection PyTypeChecker
        camera_xs, camera_ys = (np.linspace(0, w - 1, w) + 0.5) / w, (np.linspace(0, h - 1, h) + 0.5) / h
        camera_xs, camera_ys = np.meshgrid(camera_xs, camera_ys)
        camera_ps = np.vstack([camera_xs, camera_ys])
        camera_frustum = camera.create_frustum_points()

        camera_p3d = math.create_points_in_frustum(camera_ps, camera_frustum, ratio=h/w)
        # TODO: implement intersection
        return camera_p3d
