#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio

from triangulum.rendering import gl
from triangulum.rendering.entities.scene import Scene
from triangulum.rendering.entities.camera import Camera


class SimpleRenderer:

    def __init__(self, gl_executor: gl.RenderingAsyncExecutor=None):
        self._shader = None
        """:type : triangulum.rendering.gl.Shader"""
        self._gl_executor = gl_executor or gl.RenderingAsyncExecutor()

    @asyncio.coroutine
    def init(self):
        yield from self._gl_executor.init_gl_context()

    @staticmethod
    def _render(scene: Scene, camera: Camera):
        if scene.projector is not None:
            scene.projector.render_shadow(scene)
        gl.clear_viewport()
        scene.render(camera)
        gl.glFinish()

    @asyncio.coroutine
    def render(self, scene: Scene, camera: Camera):
        yield from self._gl_executor.map(self._render, scene, camera)
