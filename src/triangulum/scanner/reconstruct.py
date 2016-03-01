#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import numpy as np

from triangulum.utils import math
from triangulum.rendering.entities.camera import Camera
from triangulum.scanner.central_line_extraction import CentralLineExtractionProcessor

logger = logging.getLogger(__name__)


class ReconstructionBuilder:

    def __init__(self, line_extraction_processor: CentralLineExtractionProcessor=None):
        self._line_extraction_processor = line_extraction_processor or CentralLineExtractionProcessor()

        self._max_projector_lod = -1

        self._wh = None
        self._stripe_ids_img = None
        self._is_ok_mask = None

    def _init(self, w, h):
        assert self._stripe_ids_img is None and self._is_ok_mask is None and self._wh is None
        self._wh = (w, h)
        self._stripe_ids_img = np.zeros((h, w), np.int32)
        self._is_ok_mask = np.ones((h, w), np.bool)

    def process_observation(self, projector_lod, img):
        assert self._max_projector_lod + 1 == projector_lod
        self._max_projector_lod = projector_lod

        h, w = img.shape[:2]
        if projector_lod == 0:
            self._init(w, h)
        else:
            assert (w, h) == self._wh

        is_green, is_red, bad_mask = self._classify(img)
        self._is_ok_mask[bad_mask] = 0
        self._stripe_ids_img = self._stripe_ids_img * 2 + is_green
        logger.debug('After projector lod={}: {}% points recognized.'
                     .format(projector_lod, 100 * self._is_ok_mask.sum() // np.prod(self._is_ok_mask.shape)))

    @staticmethod
    def _classify(img):
        h, w = img.shape[:2]
        is_green = img[:, :, 1] > 128
        is_red = img[:, :, 0] > 128
        is_green, is_red = np.logical_and(is_green, ~is_red), np.logical_and(is_red, ~is_green)
        bad = np.logical_and(~is_green, ~is_red)
        return is_green, is_red, bad

    def _create_pixels_3d(self, camera, w, h):
        camera_xs, camera_ys = (np.linspace(0, w - 1, w) + 0.5) / w,\
                               (np.linspace(0, h - 1, h) + 0.5) / w
        camera_ys, camera_xs = np.meshgrid(camera_ys, camera_xs)
        camera_ps = np.hstack([camera_xs.reshape(-1, 1), camera_ys.reshape(-1, 1)])

        camera_frustum = camera.create_frustum_points()
        camera_ps3d = math.create_points_in_frustum(camera_ps, camera_frustum, ratio=h / w)
        camera_ps3d = camera_ps3d.reshape(w, h, 3)
        return camera_ps3d

    def build_point_cloud(self, projector_camera: Camera, camera: Camera):
        w, h = self._wh

        camera_ps3d = self._create_pixels_3d(camera, w, h)
        projector_ps3d = self._create_pixels_3d(projector_camera, 2 ** self._max_projector_lod, 2)

        is_central_line_pixel = self._line_extraction_processor.process(self._stripe_ids_img)
        self._is_ok_mask = self._is_ok_mask & is_central_line_pixel

        points_3d = []
        y_ids, x_ids = np.nonzero(self._is_ok_mask)
        for y_id, x_id in zip(y_ids, x_ids):  # TODO: this is bottle-neck
            ray = math.normalize(camera_ps3d[x_id, y_id] - camera.get_position())
            stripe_id = self._stripe_ids_img[y_id, x_id]
            stripe_plane = math.plane_by_points([projector_camera.get_position(),
                                                 projector_ps3d[stripe_id, 0], projector_ps3d[stripe_id, 1]])
            p3d = math.intersect_plane_line(stripe_plane, ray, camera.get_position())
            points_3d.append(p3d)
        return np.array(points_3d)
