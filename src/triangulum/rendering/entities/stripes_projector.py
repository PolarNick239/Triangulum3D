#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.rendering import gl
from triangulum.rendering.entities.abstract import Renderable
from triangulum.rendering.entities.camera import Camera

color_fp = '''
    const float EPSILON = 0.0002;
    uniform mat4 projector_mvp_mtx;
    uniform sampler2D projector_depth_tex;

    const vec4 projector_stripes_colours[{stripes_types_number}] = vec4[{stripes_types_number}](
{stripes_colours}
    );

    vec4 getProjectorColor(vec4 position) {{
        vec4 projector_position = projector_mvp_mtx * position;
        vec2 uv = projector_position.xy / projector_position.w;
        float z = projector_position.z / projector_position.w;

        if (any(lessThan(uv, vec2(-1.0 + EPSILON))) || any(greaterThan(uv, vec2(1.0 - EPSILON)))) {{
            return vec4(0.0);
        }}

        uv = (uv + 1.0) / 2.0;
        z = (z + 1.0) / 2.0;
        float projector_depth = texture(projector_depth_tex, uv).x;
        if (z < projector_depth + EPSILON) {{
            int stripe_id = int(floor(uv.x * {stripes_number})) % {stripes_types_number};
            return projector_stripes_colours[stripe_id];
        }} else {{
            return vec4(0.0);
        }}
    }}
'''


class StripesProjector(Camera):

    def __init__(self, *,
                 stripes_number=63, stripes_colours=((1, 0, 0), (0, 1, 0)),
                 course=-135, pitch=360-30, distance=5, target=(0, 0, 0),
                 aspect=1.0, fov_h=45, near=1.0, far=1000000000.0):
        super(StripesProjector, self).__init__(
            course=course, pitch=pitch, distance=distance, target=target,
            aspect=aspect, fov_h=fov_h, near=near, far=far)
        self._stripes_number = stripes_number
        if np.array(stripes_colours).shape[1] == 3:
            stripes_colours = np.hstack([stripes_colours, [[1]] * len(stripes_colours)])
        self._stripes_colours = stripes_colours

        self._initialized = False
        self._depth_map_wh = (2048, 2048)
        self._framebuffer = None
        ''':type : gl.Framebuffer'''
        self._depth_buffer = None
        ''':type : gl.Texture2D'''

        self._rendering_depth_map = False

    def get_shader_code(self):
        return color_fp.format(stripes_types_number=len(self._stripes_colours),
                               stripes_number=self._stripes_number,
                               stripes_colours=',\n'.join(['        vec4({})'.format(', '.join(map(str, colour)))
                                                           for colour in self._stripes_colours]))

    def _init(self):
        w, h = self._depth_map_wh
        self._framebuffer = gl.Framebuffer()
        self._depth_buffer = gl.create_tex(w, h, gl.GL_DEPTH_COMPONENT24)
        self._initialized = True

    def release(self):
        if self._initialized:
            self._framebuffer.release()
            self._depth_buffer.release()
            self._framebuffer = None
            self._depth_buffer = None

    def uniforms(self, shader):
        shader.uniform_matrix_f('projector_mvp_mtx', self.get_mvp_matrix())
        shader.bind_textures(projector_depth_tex=self._depth_buffer)

    def set_projector(self, projector):
        super(StripesProjector, self).set_projector(projector)

    def render(self, camera, *, edges_mode=False):
        if not self._rendering_depth_map:
            super(StripesProjector, self).render(camera, edges_mode=edges_mode)

    def add_textures(self, textures):
        textures['projector_depth_tex'] = self._depth_buffer

    def render_shadow(self, renderable: Renderable):
        if not self._initialized:
            self._init()
        self._rendering_depth_map = True
        with gl.render_to_texture(self._framebuffer, depth=self._depth_buffer,
                                  viewport_size=self._depth_map_wh):
            gl.clear_viewport()
            renderable.render(self)
            gl.glFinish()
        self._rendering_depth_map = False
