#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import logging
import colorsys

import functools
import numpy as np

from triangulum.rendering import gl
from triangulum.rendering.entities.abstract import Renderable

logger = logging.getLogger(__name__)


color_common = '''
    uniform mat4 mvp_mtx;

    V2F {
        vec4 color;
        vec4 position;
    } v2f;
'''

color_vp = '''
    in vec4 position;
    in vec4 color;

    void main(void)
    {
        gl_Position = mvp_mtx * position;
        v2f.color = color;
        v2f.position = position;
    }
'''

color_fp = '''
    layout(location=0) out vec4 out_color;

    void main(void)
    {
        out_color = apply_light(v2f.color);
    }
'''


textured_common = '''
    uniform mat4 mvp_mtx;

    V2F {
        vec2 uv;
        vec4 position;
    } v2f;
'''

textured_vp = '''
    in vec4 position;
    in vec2 uv;

    void main(void)
    {
        gl_Position = mvp_mtx * position;
        v2f.uv = uv;
        v2f.position = position;
    }
'''

textured_fp = '''
    uniform sampler2D tex;

    layout(location=0) out vec4 out_color;

    void main(void)
    {
        out_color = apply_light(texture(tex, v2f.uv));
    }
'''

ambient_light_fp = '''
    vec4 apply_light(vec4 color)
    {
        return color;
    }
'''

projector_light_fp = '''
    uniform mat4 projector_mvp_mtx;

    const float ambient = 0.5;

    vec4 apply_light(vec4 color)
    {
        vec4 projector_position = projector_mvp_mtx * v2f.position;
        projector_position /= projector_position.w;
        vec4 color_with_light = color * ambient;

        if (true) {
            vec2 uv = projector_position.xy;
            vec4 projector_color = getProjectorColor(uv);
            color_with_light += projector_color * (1.0 - ambient);
        }
        return color_with_light;
    }
'''


def get_shader(*, is_textured, projector_shader_fp=None):
    light_fp = projector_shader_fp + projector_light_fp if projector_shader_fp else ambient_light_fp
    if is_textured:
        return gl.Shader(textured_common + textured_vp, textured_common + light_fp + textured_fp, 400, 400)
    else:
        return gl.Shader(color_common + color_vp, color_common + light_fp + color_fp, 400, 400)


class PointsCloud(Renderable):

    def __init__(self, verts, *, colors=True, uv=None, faces=None, edges=None, name=None):
        if colors is True:
            if uv is None:
                colors = auto_color(verts)
            else:
                colors = None
        assert colors is None or uv is None

        self.verts = verts
        self.colors = colors
        self.uv = uv
        self.texture = None
        self.faces = faces
        self.edges = edges
        self.name = name or '{} points'.format(len(verts))

        if self.uv is None:
            self._attributes = dict(position=self.verts, color=self.colors)
        else:
            self._attributes = dict(position=self.verts, uv=self.uv)
        self._shader_constructor = functools.partial(get_shader, is_textured=self.uv is not None)
        self._shader = None
        self._projector = None
        ''':type : StripesProjector'''

    def dump_ply(self, filename):
        if self.edges is not None:
            logger.error('Saving point cloud with {} edges! But dump_ply does not support edges!'
                         .format(len(self.edges)))
        dump_ply(filename, self.verts, colors=self.colors, uv=self.uv, faces=self.faces)

    def set_texture(self, texture: gl.Texture2D):
        assert self.uv is not None
        self.texture = texture

    def set_projector(self, projector):
        self._shader_constructor = functools.partial(get_shader, is_textured=self.uv is not None,
                                                     projector_shader_fp=projector.get_shader_code())
        self._shader = None
        self._projector = projector

    def render(self, camera,
               *, edges_mode=False):
        if self._shader is None:
            self._shader = self._shader_constructor()
        shader = self._shader

        with shader:
            mvp_mtx = camera.get_mvp_matrix()
            shader.uniform_matrix_f('mvp_mtx', mvp_mtx)
            if self._projector:
                shader.uniform_matrix_f('projector_mvp_mtx', self._projector.get_mvp_matrix())

            if self.uv is not None:
                shader.bind_textures(tex=self.texture)

            if self.faces is not None:
                if not edges_mode:
                    gl.draw_attrib_arrays(gl.GL_TRIANGLES, indices=np.int32(self.faces), **self._attributes)
                else:
                    edges_indices = np.vstack([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [0, 2]]])
                    gl.draw_attrib_arrays(gl.GL_LINES, indices=edges_indices, **self._attributes)
            if self.edges is not None:
                gl.draw_attrib_arrays(gl.GL_LINES, indices=np.int32(self.edges), **self._attributes)

            if self.faces is None and self.edges is None:
                gl.draw_attrib_arrays(gl.GL_POINTS, **self._attributes)


def load_ply(filename) -> PointsCloud:
    file = open(filename, 'rb')
    prop_to_dtype = {'float': np.float32, 'int': np.int32, 'uchar': np.uint8}

    header = []
    while True:
        splitted_byte_line = file.readline().split()
        splitted_line = []
        for bytez in splitted_byte_line:
            splitted_line.append(bytez.decode())
        if len(splitted_line) == 1 and splitted_line[0] == 'ply':
            continue
        header.append(splitted_line)
        if splitted_line[0] == 'end_header':
            break

    it = iter(header)
    splitted_line = it.__next__()
    elements = {}
    while True:
        if splitted_line[0] == 'end_header':
            break
        if splitted_line[0] == 'element':
            element_name, element_len = splitted_line[1], int(splitted_line[2])
            element_props = []
            splitted_line = it.__next__()
            while splitted_line[0] == 'property':
                element_props.append(splitted_line)
                splitted_line = it.__next__()
            if element_name == 'face':
                el_type = np.dtype([('count', np.uint8), ('idx', np.int32, 3)])
                elements[element_name] = np.fromfile(file, el_type, element_len)['idx'].copy()
            else:
                el_type = np.dtype([(name, np.dtype(prop_to_dtype[tp])) for _, tp, name in element_props])
                elements[element_name] = np.fromfile(file, el_type, element_len)
            continue
        splitted_line = it.__next__()

    v = elements['vertex']
    xyz, rgb, nxyz, texture_uv = ['x', 'y', 'z'], ['red', 'green', 'blue'],\
                                 ['nx', 'ny', 'nz'], ['texture_u', 'texture_v']
    if set(v.dtype.fields).issuperset(xyz):
        elements['xyz'] = v[xyz].view((np.float32, 3))
    if set(v.dtype.fields).issuperset(nxyz):
        elements['normal'] = v[nxyz].view((np.float32, 3))
    if set(v.dtype.fields).issuperset(rgb):
        elements['rgb'] = v[rgb].view((np.uint8, 3))
    if set(v.dtype.fields).issuperset(texture_uv):
        elements['texture_uv'] = v[texture_uv].view((np.float32, 2))
    for key in ['xyz', 'rgb', 'face']:
        if key not in elements:
            elements[key] = None if not key == 'rgb' else True
    return PointsCloud(elements['xyz'], colors=elements['rgb'], uv=elements['texture_uv'], faces=elements['face'],
                       name=filename.split('/')[-1].split('.')[0])


def auto_color(verts):
    if len(verts) == 0:
        return None
    z = verts[:, 2]
    min_z, max_z = np.percentile(z, 5), np.percentile(z, 95)
    if max_z == min_z:
        zs01 = np.zeros_like(z)
    else:
        zs01 = np.clip((z - min_z) / (max_z - min_z), 0, 1)

    # HSV
    colors = np.zeros((len(zs01), 3))
    for i, z01 in enumerate(zs01):
        colors[i] = colorsys.hsv_to_rgb(1 - z01, 0.75, 0.2 + 0.8 * (z01 ** 0.5))

    return colors


def dump_ply(fn, verts, colors=True, normals=None, uv=None, faces=None,
             extra_verts=None, extra_colors=None, extra=None):
    if extra is None:
        extra = []
    header = ['ply', 'format binary_little_endian 1.0'] + extra

    verts = verts.reshape(-1, 3)

    if colors is True:
        colors = auto_color(verts)
    elif colors is False:
        colors = None

    if extra_verts is not None:
        extra_verts = extra_verts.reshape(-1, 3)
        verts = np.vstack([verts, extra_verts])
        if colors is not None:
            if extra_colors is None:
                extra_colors = np.zeros_like(extra_verts)
                extra_colors[:, 0] = 255
            colors = np.vstack([colors, extra_colors])

    vert_t = [('vert', np.float32, 3)]
    header += ['element vertex %d' % len(verts)]
    header += ['property float %s' % s for s in 'xyz']
    fields = {'vert': verts}

    if colors is not None:
        fields['color'] = np.asarray(colors).reshape(-1, 3)
        assert len(fields['color']) == len(verts)
        vert_t.append(('color', np.uint8, 3))
        header += ['property uchar %s' % s for s in ['red', 'green', 'blue']]
    if normals is not None:
        fields['normal'] = np.asarray(normals).reshape(-1, 3)
        assert len(fields['normal']) == len(verts)
        vert_t.append(('normal', np.float32, 3))
        header += ['property float %s' % s for s in ['nx', 'ny', 'nz']]
    if uv is not None:
        fields['uv'] = np.asarray(uv).reshape(-1, 2)
        assert len(fields['uv']) == len(verts)
        vert_t.append(('uv', np.float32, 2))
        header += ['property float %s' % s for s in ['texture_u', 'texture_v']]

    vertex_data = np.zeros(len(verts), np.dtype(vert_t))
    for name in fields:
        vertex_data[name] = fields[name]

    face_data = None
    if faces is not None:
        faces = np.int32(faces).reshape(-1, 3)
        face_t = np.dtype([('vn', np.uint8), ('vs', np.uint32, 3)])
        face_data = np.zeros(len(faces), face_t)
        face_data['vn'][:] = 3
        face_data['vs'] = faces
        header += ["element face %d" % len(face_data)]
        header += ["property list uchar int vertex_indices"]
    header += ["end_header"]

    with open(fn, 'wb') as f:
        f.writelines(map(lambda s: bytes(s, 'UTF-8'), '\n'.join(header) + '\n'))
        vertex_data.tofile(f)
        if face_data is not None:
            face_data.tofile(f)
