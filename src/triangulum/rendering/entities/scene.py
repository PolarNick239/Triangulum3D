#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.rendering.entities.abstract import Renderable
from triangulum.rendering.entities.points_cloud import PointsCloud
from triangulum.rendering.entities.camera import Camera
from triangulum.rendering.entities.stripes_projector import StripesProjector


class Scene(Renderable):

    def __init__(self, show_axis=True, edges_mode=False):
        self._show_axis = show_axis
        self._edges_mode = edges_mode

        self._axis_vectors = self._create_axis_vectors()
        ''':type : PointsCloud'''
        self._renderables = []
        ''':type : list[Renderable]'''
        self._projector = None
        ''':type : StripesProjector'''

    @property
    def show_axis(self):
        return self._show_axis

    @show_axis.setter
    def show_axis(self, show_axis):
        self._show_axis = show_axis

    @property
    def edges_mode(self):
        return self._edges_mode

    @edges_mode.setter
    def edges_mode(self, edges_mode):
        self._edges_mode = edges_mode

    def add_renderable(self, renderable: Renderable):
        self._renderables.append(renderable)

    def set_projector(self, projector: StripesProjector=None):
        self._projector = projector
        for renderable in self._renderables:
            renderable.set_projector(projector)
        self._projector.set_projector(projector)

    def render(self, camera: Camera,
               *, edges_mode=False):
        if self._show_axis:
            self._axis_vectors.render(camera, edges_mode=edges_mode)
        for renderable in self._renderables:
            renderable.render(camera, edges_mode=edges_mode)
        self._projector.render(camera, edges_mode=edges_mode)

    @staticmethod
    def _create_axis_vectors() -> PointsCloud:
        points = np.array([
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], np.float32)
        colors = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], np.float32)
        edges = np.array([[0, 1], [0, 2], [0, 3]], np.int32)
        return PointsCloud(verts=points, colors=colors, edges=edges)
