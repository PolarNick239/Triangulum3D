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


class ImageRenderer:

    def __init__(self, gl_executor: gl.RenderingAsyncExecutor):
        self._gl_executor = gl_executor or gl.RenderingAsyncExecutor()

        self._framebuffer = None
        self._color_buffer = None
        self._depth_buffer = None

        self._initialized_wh = None

    @asyncio.coroutine
    def _update_resources(self, viewport_size_wh):
        if viewport_size_wh != self._initialized_wh:
            def update_resources():
                if self._framebuffer is None:
                    self._framebuffer = gl.Framebuffer()
                w, h = viewport_size_wh
                gl.release([self._color_buffer, self._depth_buffer])
                self._color_buffer = gl.create_tex(w, h, gl.GL_RGBA8)
                self._depth_buffer = gl.create_tex(w, h, gl.GL_DEPTH_COMPONENT32)
            yield from self._gl_executor.map(update_resources)
            self._initialized_wh = viewport_size_wh

    def _render(self, scene: Scene, camera: Camera, viewport_size_wh):
        if scene.projector is not None:
            scene.projector.render_shadow(scene)
        with gl.render_to_texture(self._framebuffer,
                                  color=self._color_buffer, depth=self._depth_buffer,
                                  viewport_size=viewport_size_wh):
            gl.clear_viewport()
            scene.render(camera)
            gl.glFinish()
            depth = gl.read_depth(self._depth_buffer, viewport_size_wh)
            color = gl.read_color(self._color_buffer, viewport_size_wh)
        return color, depth

    @asyncio.coroutine
    def release(self):
        yield from self._gl_executor.map(gl.release, [self._framebuffer, self._color_buffer, self._depth_buffer])

    @asyncio.coroutine
    def render(self, scene: Scene, camera: Camera, viewport_size_wh):
        yield from self._update_resources(viewport_size_wh)
        color, depth = yield from self._gl_executor.map(self._render, scene, camera, viewport_size_wh)
        return color, depth
