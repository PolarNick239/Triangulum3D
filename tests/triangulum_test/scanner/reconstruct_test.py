#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio

from triangulum.utils import support
from triangulum_test.test_support import TestBase
from triangulum.rendering.entities.box import Box
from triangulum.rendering.entities.scene import Scene
from triangulum.rendering.entities.camera import Camera
from triangulum.scanner.reconstruct import ReconstructionBuilder
from triangulum.rendering.renderers.simple_renderer import ImageRenderer
from triangulum.rendering.entities.stripes_projector import StripesProjector


class ReconstructionBuilderTest(TestBase):

    def render(self, scene: Scene, camera: Camera, viewport):
        @asyncio.coroutine
        def render():
            renderer = ImageRenderer()
            color, depth = yield from renderer.render(scene, camera, viewport)
            yield from renderer.release()
            return color, depth

        color, depth = asyncio.get_event_loop().run_until_complete(render())
        return color, depth

    def reconstruction_test(self):
        scene = Scene()
        scene.add_renderable(Box([-1, 1], [-0.5, 0.5], [0, 1]))
        camera = Camera(course=40)
        viewport = (1024, 1024)
        projector_lods = 8

        color, depth = self.render(scene, camera, viewport)
        self.dump_debug_img("scene_color.png", color)
        self.dump_debug_img("scene_depth.png", support.array_to_grayscale(depth))

        reconstructor = ReconstructionBuilder()
        for i in range(projector_lods):
            projector = StripesProjector(course=70, stripes_number=2**(i + 1))
            self.register_releasable(projector)
            scene.set_projector(projector)

            color, depth = self.render(scene, camera, viewport)
            self.dump_debug_img("{}_color.png".format(i), color)

            reconstructor.process_observation(i, color)

        reconstructor.build_point_cloud(projector, camera)
        # TODO: implement result check
