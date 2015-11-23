#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np
from enum import Enum

from triangulum.utils import math
from triangulum.utils import aabb
from triangulum.rendering.entities.abstract import Renderable
from triangulum.rendering.entities.points_cloud import PointsCloud


class CameraMode(Enum):

    perspective = 'perspective'
    ortho = 'ortho'


class Camera(Renderable):

    def __init__(self,
                 *, course=-135, pitch=360-30, distance=5, target=(0, 0, 0),
                 aspect=1.0, fov_h=45, near=0.1, far=100000.0, mode: CameraMode=CameraMode.perspective):
        self.course = course
        """ Course in degrees, 0 - heading along the Y axis, clockwise rotation (looking down, like Gods!) """
        self.pitch = pitch
        """ Pitch in degrees,
            0 - heading horizontally (parallel XY plane, along course),
            (0 .. 180) - heading up,
            180 - heading horizontally backwards,
            (180 .. 360) - looking down,
            360-90 - looking along -Z axis """

        self.max_pitch = 360-0
        self.min_pitch = 360-90
        assert self.min_pitch <= self.pitch <= self.max_pitch

        self.target = np.array(target, np.float32)
        self.distance = distance

        self.mode = mode
        self.aspect = aspect

        self.near = near
        self.far = far
        self.fov_h = fov_h

        self._frustum_mesh = None
        ''':type : triangulum.rendering.entities.points_cloud.PointsCloud'''
        self._projector = None
        self._update_frustum_mesh()

    def _update_frustum_mesh(self):
        # TODO: update on every camera movement
        # (currently this is not a problem, because only StripesProjector is rendered)
        self._frustum_mesh = PointsCloud(self.create_frustum_points(),
                                         faces=[[1, 2, 3], [1, 3, 4]],
                                         edges=[[0, i] for i in range(1, 5)])

    def set_projector(self, projector):
        self._projector = projector
        self._frustum_mesh.set_projector(projector)

    def render(self, camera,
               *, edges_mode=False):
        self._frustum_mesh.render(camera, edges_mode=edges_mode)

    def _get_camera_direction(self):
        course = np.radians([self.course])[0]
        pitch = np.radians([self.pitch])[0]
        if pitch == 270:
            return np.array([0, 0, -1])
        v = np.array([0, np.cos(pitch), np.sin(pitch)])
        v = np.dot(math.rotate_matrix2d(-course), v)
        return v

    @staticmethod
    def _get_direction(course):
        course = np.radians(course)
        return np.array([np.sin(course), np.cos(course), 0])

    def rotate(self, course, pitch):
        self.course = (self.course + course) % 360

        self.pitch = self.pitch + pitch
        self.pitch = np.clip(self.pitch, self.min_pitch, self.max_pitch)

    def move(self, x, y):
        """
        :param x: axis going to right from camera
        :param y: axis going with XY projection of camera direction
        """
        direction = self._get_direction(self.course)
        right = self._get_direction(self.course + 90)
        x, y = (right[:2] * x + direction[:2] * y) * self.get_viewport_width()
        self.target[:2] += [x, y]

    def zoom_in(self, coef):
        """
        :param coef: if positive - zoom in, if negative - zoom out, if zero - nothing changes
        """
        self.distance /= 1.1 ** coef

    def set_aspect(self, aspect):
        self.aspect = aspect

    def get_mv_matrix(self):
        camera_position = self.target - math.normalize(self._get_camera_direction()) * self.distance
        return math.look_at_matrix(camera_position, self.target,
                                   up=[0, 0, 1], right=self._get_direction(self.course + 90))

    def get_viewport_width(self):
        width = 2 * np.tan(np.radians([self.fov_h])[0] / 2) * self.distance
        return width

    def get_p_matrix(self):
        if self.mode is CameraMode.perspective:
            return math.perspective_matrix(self.aspect, self.near, self.far, self.fov_h)
        else:
            assert self.mode is CameraMode.ortho
            return math.ortho_matrix(self.aspect, self.near, self.far, self.get_viewport_width())

    # noinspection PyPep8Naming
    def get_mvp_matrix(self):
        P = self.get_p_matrix()
        MV = self.get_mv_matrix()
        return np.dot(P, MV)

    def create_frustum_points(self, frustums_depth=1.0):
        rt_inv = np.linalg.inv(self.get_mv_matrix())
        p = self.get_p_matrix()[:-1, :-1]
        camera_corners = math.homo_translate(np.linalg.inv(p), aabb.rect_to_quad([[-1.0, -1.0 * self.aspect],
                                                                                  [1.0, 1.0 * self.aspect]]))
        corners = np.hstack([camera_corners, [[-1]] * 4]) * frustums_depth
        frustum_points = math.homo_translate(rt_inv, np.vstack([[[0, 0, 0]], corners]))
        return frustum_points
