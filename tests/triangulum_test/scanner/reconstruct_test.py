#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio
from pathlib import Path

from triangulum.utils import ply
from triangulum.utils import support
from triangulum_test.test_support import TestBase
from triangulum.rendering.entities.box import Box
from triangulum.rendering.entities.scene import Scene
from triangulum.rendering.entities.camera import Camera
from triangulum.scanner.reconstruct import ReconstructionBuilder
from triangulum.rendering.renderers.simple_renderer import ImageRenderer
from triangulum.rendering.entities.stripes_projector import StripesProjector
from triangulum.scanner.central_line_extraction import CentralLineExtractionProcessor


class ReconstructionBuilderTest(TestBase):

    def render(self, scene: Scene, camera: Camera, viewport):
        @asyncio.coroutine
        def render():
            renderer = ImageRenderer(self.get_gl_executor())
            color, depth = yield from renderer.render(scene, camera, viewport)
            yield from renderer.release()
            return color, depth

        color, depth = asyncio.get_event_loop().run_until_complete(render())
        return color, depth

    def reconstruction_test(self):
        scene = Scene()
        scene.add_renderable(Box([-1, 1], [-0.5, 0.5], [0, 1]))
        camera = Camera(course=40)
        viewport = (500, 300)
        projector_lods = 8

        color, depth = self.render(scene, camera, viewport)
        self.dump_debug_img("scene_color.png", color)
        self.dump_debug_img("scene_depth.png", support.array_to_grayscale(depth))

        central_line_processor = CentralLineExtractionProcessor(debug_enabled=self.with_debug_output())
        reconstructor = ReconstructionBuilder(line_extraction_processor=central_line_processor)
        projector = StripesProjector(course=70)
        self.register_releasable(projector)
        for i in range(projector_lods):
            projector.stripes_number = 2 ** i
            scene.set_projector(projector)

            color, depth = self.render(scene, camera, viewport)
            self.dump_debug_img("{}_color.png".format(i), color)

            reconstructor.process_observation(i, color)

        points_3d = reconstructor.build_point_cloud(projector, camera)

        if self.with_debug_output():
            subdir = Path('central_line')
            self.dump_debug_matrix_by_hue(subdir / 'is_edge_pixel.png', central_line_processor._debug_last_values['is_edge_pixel'])
            self.dump_debug_matrix_by_hue(subdir / 'distance.png', central_line_processor._debug_last_values['distance'])
            self.dump_debug_matrix_by_hue(subdir / 'is_maximum.png', central_line_processor._debug_last_values['is_maximum'])

            ply.write_ply(self.debug_dir() / 'points.ply', points_3d)
        # TODO: implement result check
