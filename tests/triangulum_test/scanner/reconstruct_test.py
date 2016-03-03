#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import asyncio
import unittest
from pathlib import Path

from triangulum.rendering.entities.points_cloud import PointsCloud
from triangulum.utils import ply
from triangulum.utils import support
from triangulum_test import test_support
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

    def box_test(self):
        scene = Scene()
        scene.add_renderable(Box([-1, 1], [-0.5, 0.5], [0, 1]))
        camera = Camera(course=40)
        projector = StripesProjector(course=70)
        viewport = (500, 300)
        projector_lods = 8

        self.reconstruction_case('box', scene, camera, projector, viewport, projector_lods)

    @unittest.skip("Too slow because of bottle-neck in ReconstructionBuilder.build_point_cloud(...)")
    def bunny_test(self):
        bunny = ply.read_ply(test_support.resources_dir_path / 'scenes' / 'bunny' / 'bunny.ply')

        def transform_bunny(ps):
            world_pos = [0.5, -0.2, 0.0]
            ps[:, 0] += world_pos[0]
            ps[:, [0, 1, 2]] = ps[:, [0, 2, 1]]
            ps = world_pos + (ps - world_pos) * 4
            return ps

        scene = Scene()
        scene.add_renderable(PointsCloud(transform_bunny(bunny['xyz']).copy(), faces=bunny['face']))
        camera = Camera(course=-120)
        projector = StripesProjector()
        viewport = (5616, 3744)
        projector_lods = 10
        self.reconstruction_case('bunny', scene, camera, projector, viewport, projector_lods)

    def reconstruction_case(self, scene_name, scene, camera, projector, viewport, projector_lods):
        color, depth = self.render(scene, camera, viewport)
        self.dump_debug_img(Path(scene_name) / "scene_color.png", color)
        self.dump_debug_img(Path(scene_name) / "scene_depth.png", support.array_to_grayscale(depth))

        central_line_processor = CentralLineExtractionProcessor(debug_enabled=self.with_debug_output())
        reconstructor = ReconstructionBuilder(line_extraction_processor=central_line_processor)
        self.register_releasable(projector)
        for i in range(projector_lods):
            projector.stripes_number = 2 ** i
            scene.set_projector(projector)

            color, depth = self.render(scene, camera, viewport)
            self.dump_debug_img(Path(scene_name) / "{}_color.png".format(i), color)

            reconstructor.process_observation(i, color)

        points_3d = reconstructor.build_point_cloud(projector, camera)

        if self.with_debug_output():
            subdir = Path(scene_name) / 'central_line'
            self.dump_debug_matrix_by_hue(subdir / 'is_edge_pixel.png', central_line_processor._debug_last_values['is_edge_pixel'])
            self.dump_debug_matrix_by_hue(subdir / 'distance.png', central_line_processor._debug_last_values['distance'])
            self.dump_debug_matrix_by_hue(subdir / 'is_maximum.png', central_line_processor._debug_last_values['is_maximum'])

            ply.write_ply(self.debug_dir() / scene_name / 'points.ply', points_3d)
        # TODO: implement result check
