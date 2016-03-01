#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import ctypes
import asyncio
import logging
import threading
import contextlib
import numpy as _np
from PIL.Image import open as image_open
from ctypes import byref
from abc import ABCMeta, abstractmethod

import OpenGL

OpenGL.FULL_LOGGING = False
OpenGL.ERROR_CHECKING = True

from OpenGL.GL.VERSION.GL_1_0 import GLenum, GLuint, glTexImage2D, glViewport, glEnable, glDisable, glFinish, glReadPixels, glGetTexImage
from OpenGL.GL.VERSION.GL_1_0 import GL_UNSIGNED_BYTE, GL_SHORT, GL_INT, GL_UNSIGNED_INT, GL_FLOAT, GL_TRUE, GL_FALSE
from OpenGL.GL.VERSION.GL_1_0 import glTexParameteri, glTexParameterf, glTexParameterfv, glTexParameteriv, glPixelStorei, glGetIntegerv
from OpenGL.GL.VERSION.GL_1_0 import glLineWidth, glPointSize, glGetIntegerv
from OpenGL.GL.VERSION.GL_1_1 import GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_PIXELS, GL_PACK_ALIGNMENT, GL_UNPACK_ALIGNMENT
from OpenGL.GL.VERSION.GL_1_1 import glBindTexture, glGenTextures, glDeleteTextures, glTexSubImage2D, GL_TEXTURE_1D, GL_TEXTURE_2D
from OpenGL.GL.VERSION.GL_1_1 import GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER
from OpenGL.GL.VERSION.GL_1_1 import GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_S, GL_TEXTURE_BORDER_COLOR
from OpenGL.GL.VERSION.GL_1_1 import GL_LINEAR, GL_NEAREST, GL_RGB, GL_RGBA, GL_RGBA8, GL_RED, GL_UNSIGNED_SHORT, GL_R
from OpenGL.GL.VERSION.GL_1_1 import GL_LINE_SMOOTH, GL_POINT_SMOOTH, GL_VIEWPORT
from OpenGL.GL.VERSION.GL_1_1 import glClear, glClearColor, glClearDepth, GL_COLOR_BUFFER_BIT, GL_TRIANGLES, GL_LINES, GL_POINTS, GL_DEPTH_TEST, GL_DEPTH_BUFFER_BIT, GL_DEPTH_COMPONENT
from OpenGL.GL.VERSION.GL_1_1 import glDrawElements, glDrawArrays, GL_TRIANGLE_FAN
from OpenGL.GL.VERSION.GL_1_2 import glTexSubImage3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE, GL_TEXTURE_MAX_LEVEL, GL_TEXTURE_BASE_LEVEL
from OpenGL.GL.VERSION.GL_1_3 import glActiveTexture, GL_TEXTURE0, GL_CLAMP_TO_BORDER
from OpenGL.GL.VERSION.GL_1_4 import GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT32
from OpenGL.GL.VERSION.GL_1_5 import glGenBuffers, glDeleteBuffers, glBindBuffer, glMapBuffer, glUnmapBuffer, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER
from OpenGL.GL.VERSION.GL_1_5 import GL_STATIC_DRAW, glBufferData, GL_WRITE_ONLY, GL_STREAM_DRAW
from OpenGL.GL.VERSION.GL_2_0 import glCreateProgram, glCreateShader, glUseProgram, glShaderSource
from OpenGL.GL.VERSION.GL_2_0 import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_CURRENT_PROGRAM
from OpenGL.GL.VERSION.GL_2_0 import glCompileShader, glAttachShader, glLinkProgram, GL_COMPILE_STATUS, GL_LINK_STATUS
from OpenGL.GL.VERSION.GL_2_0 import glVertexAttrib1f, glVertexAttrib2f, glVertexAttrib3f, glVertexAttrib4f
from OpenGL.GL.VERSION.GL_2_0 import glGetShaderiv, glGetProgramiv, glGetShaderInfoLog, glGetProgramInfoLog
from OpenGL.GL.VERSION.GL_2_0 import glVertexAttribPointer, glEnableVertexAttribArray, glDisableVertexAttribArray
from OpenGL.GL.VERSION.GL_2_0 import glGetAttribLocation, glGetUniformLocation, glDrawBuffers
from OpenGL.GL.VERSION.GL_2_0 import glUniformMatrix2fv, glUniformMatrix3fv, glUniformMatrix4fv
from OpenGL.GL.VERSION.GL_2_1 import GL_PIXEL_UNPACK_BUFFER
from OpenGL.GL.VERSION.GL_3_0 import GL_FRAMEBUFFER, glGenFramebuffers, glBindFramebuffer, glDeleteFramebuffers
from OpenGL.GL.VERSION.GL_3_0 import glFramebufferTexture2D, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT
from OpenGL.GL.VERSION.GL_3_0 import glCheckFramebufferStatus, GL_FRAMEBUFFER_COMPLETE, GL_TEXTURE_2D_ARRAY
from OpenGL.GL.VERSION.GL_3_0 import GL_RGBA16I, GL_RGBA16F, GL_RG16F, GL_RGBA32F, GL_R32F, GL_RG32F, GL_RGBA_INTEGER, GL_HALF_FLOAT, GL_RG, GL_RG8
from OpenGL.GL.VERSION.GL_3_1 import glDrawElementsInstanced, glDrawArraysInstanced
from OpenGL.GL.VERSION.GL_3_3 import glVertexAttribDivisor
from OpenGL.GL.VERSION.GL_4_1 import glProgramUniform1iv, glProgramUniform2iv, glProgramUniform3iv, glProgramUniform4iv, glProgramUniform1i
from OpenGL.GL.VERSION.GL_4_1 import glProgramUniform1fv, glProgramUniform2fv, glProgramUniform3fv, glProgramUniform4fv
from OpenGL.GL.VERSION.GL_4_2 import glTexStorage2D, glTexStorage3D

from OpenGL.GL.EXT.direct_state_access import glNamedBufferDataEXT
from OpenGL.GL.ARB.pixel_buffer_object import GL_PIXEL_UNPACK_BUFFER_ARB

from triangulum.utils.support import AsyncExecutor
from triangulum.rendering.context.gl_context import GLContext


logger = logging.getLogger(__name__)

try:
    from triangulum.rendering.context.glfw_context import GLFWContext as GLContextImpl
except Exception as e:
    logger.error('GLFWContext import failed!')
    GLContextImpl = None


_gl_thread_local = threading.local()


def _array_gl_type(array):
    return dict(
        uint8=GL_UNSIGNED_BYTE,
        int16=GL_SHORT,
        int32=GL_INT,
        uint32=GL_UNSIGNED_INT,
        float32=GL_FLOAT)[array.dtype.name]


class RenderingAsyncExecutor(AsyncExecutor):

    def __init__(self, gl_context: GLContext=None, loop=None):
        super().__init__(1, loop or asyncio.get_event_loop())
        self._gl_context = gl_context
        self._initialized = False

    def _init_gl_context(self):
        if self._gl_context is None:
            self._gl_context = GLContextImpl()
        _gl_thread_local.IS_GL_THREAD = True
        self._initialized = True

    @asyncio.coroutine
    def init_gl_context(self):
        if not self._initialized:
            yield from super(RenderingAsyncExecutor, self).map(self._init_gl_context)

    def _with_activated_context(self, fn, *args):
        with self._gl_context:
            return fn(*args)

    @asyncio.coroutine
    def map(self, fn, *args):
        yield from self.init_gl_context()
        return (yield from super(RenderingAsyncExecutor, self).map(self._with_activated_context, fn, *args))


class GLReleasable(metaclass=ABCMeta):

    def __init__(self):
        assert getattr(_gl_thread_local, 'IS_GL_THREAD', False)
        self._resource_thread_id = threading.current_thread().ident
        self.released = False

    def __del__(self):
        assert self.released

    def release(self):
        assert self._resource_thread_id == threading.current_thread().ident
        self._release()
        self.released = True

    @abstractmethod
    def _release(self):
        pass


def release(releasable_list):
    for releasable in releasable_list:
        if releasable is not None:
            releasable.release()


class _Bindable(metaclass=ABCMeta):

    def __init__(self):
        self._bound = False

    def __enter__(self):
        self._bound = True
        self.bind()

    def __exit__(self, *_):
        self.unbind()
        self._bound = False

    @abstractmethod
    def bind(self):
        pass

    @abstractmethod
    def unbind(self):
        pass


class _BufferObject(_Bindable, GLReleasable):

    target = None

    def __init__(self, data=None, usage=GL_STATIC_DRAW):
        _Bindable.__init__(self)
        GLReleasable.__init__(self)
        self.handle = GLuint()
        glGenBuffers(1, byref(self.handle))
        self._as_parameter_ = self.handle

        self.shape = None
        if data is not None:
            data = _np.ascontiguousarray(data)
            with self:
                glBufferData(self.target, data.size * data.itemsize, data, usage)

    def set_size(self, size, usage):
        glNamedBufferDataEXT(self.handle, size, ctypes.c_void_p(0), usage)

    def bind(self):
        glBindBuffer(self.target, self.handle)

    def unbind(self):
        glBindBuffer(self.target, 0)

    def _release(self):
        glDeleteBuffers(1, byref(self._as_parameter_))


class ArrayBuffer(_BufferObject):
    target = GL_ARRAY_BUFFER


class ElementArrayBuffer(_BufferObject):
    target = GL_ELEMENT_ARRAY_BUFFER


class PixelBufferObject(_BufferObject):
    target = GL_PIXEL_UNPACK_BUFFER_ARB


class BufferMapper:
    def __init__(self, buffer, usage):
        self._target = buffer.target
        self._usage = usage

    def __enter__(self):
        return glMapBuffer(self._target, self._usage)

    def __exit__(self, *_):
        glUnmapBuffer(self._target)


class _Texture(_Bindable, GLReleasable):

    target = None

    def __init__(self):
        _Bindable.__init__(self)
        GLReleasable.__init__(self)
        self._handle = glGenTextures(1)
        self._as_parameter_ = self._handle

    def bind(self):
        glBindTexture(self.target, self)

    def unbind(self):
        glBindTexture(self.target, 0)

    def _release(self):
        glDeleteTextures(1, _np.uint32([self._handle]))

    def set_params(self, args):
        assert self._bound
        for param_name, val in args:
            if isinstance(val, int):
                glTexParameteri(self.target, param_name, val)
            elif isinstance(val, float):
                glTexParameterf(self.target, param_name, val)
            elif isinstance(val[0], int):
                glTexParameteriv(self.target, param_name, _np.array(val, _np.int32))
            elif isinstance(val[0], float):
                glTexParameterfv(self.target, param_name, _np.array(val, _np.float32))
            else:
                raise Exception('Value of unexpected type: {}. (for parameter {})'.format(val, param_name))


class Texture1D(_Texture):
    target = GL_TEXTURE_1D


class Texture2D(_Texture):
    target = GL_TEXTURE_2D


class Texture2DArray(_Texture):
    target = GL_TEXTURE_2D_ARRAY


NO_TEXTURE_HANDLE = 0


LINEAR_LINEAR = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)]
NEAREST_NEAREST = [(GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST)]
CLAMP_TO_EDGE = [(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                 (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]
CLAMP_TO_BORDER = [(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER),
                   (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)]
NO_MIPMAPING = [(GL_TEXTURE_BASE_LEVEL, 0),
                (GL_TEXTURE_MAX_LEVEL, 0)]


def create_tex(w, h, internal_format, params=NEAREST_NEAREST) -> Texture2D:
    tex = Texture2D()
    with tex:
        tex.set_params(params)
        glTexStorage2D(tex.target, 1, internal_format, w, h)
    return tex


def read_texture(texture, size, channels, gl_format, dtype):
    w, h = size
    if channels == 1:
        img = _np.zeros((h, w), dtype)
    else:
        img = _np.zeros((h, w, channels), dtype)
    with texture:
        glGetTexImage(texture.target, 0, gl_format, _array_gl_type(img), img)
    img = img[::-1]
    return img


def read_color(texture, size):
    return read_texture(texture, size, 4, GL_RGBA, _np.uint8)


def read_depth(texture, size):
    return read_texture(texture, size, 1, GL_DEPTH_COMPONENT, _np.float32)


class Framebuffer(_Bindable, GLReleasable):

    def __init__(self):
        _Bindable.__init__(self)
        GLReleasable.__init__(self)
        self._handle = glGenFramebuffers(1)
        self._as_parameter_ = self._handle

    def _release(self):
        glDeleteFramebuffers(1, _np.uint32([self._handle]))

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


@contextlib.contextmanager
def render_to_texture(framebuffer, color=None, depth=None, viewport_size=None):
    if color is None:
        color = []
    with framebuffer:
        if not isinstance(color, list):
            color = [color]
        draw_bufs = []
        for i, tex in enumerate(color):
            if tex is None:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, NO_TEXTURE_HANDLE, 0)
            else:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, tex.target, tex, 0)
            draw_bufs.append(GL_COLOR_ATTACHMENT0 + i)
        enum_array_n = GLenum * len(draw_bufs)
        draw_bufs = enum_array_n(*draw_bufs)

        glDrawBuffers(len(draw_bufs), draw_bufs)
        if depth is None:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, NO_TEXTURE_HANDLE, 0)
        else:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth.target, depth, 0)

        if viewport_size is not None:
            w, h = viewport_size
            old_viewport_xywh = glGetIntegerv(GL_VIEWPORT)
            glViewport(0, 0, w, h)
        else:
            old_viewport_xywh = None

        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        yield

        for i, _ in enumerate(color):
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, NO_TEXTURE_HANDLE, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, NO_TEXTURE_HANDLE, 0)

        if old_viewport_xywh is not None:
            glViewport(*old_viewport_xywh)


def clear_viewport():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glClearDepth(1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


@contextlib.contextmanager
def enable_states(*state):
    for idx in state:
        glEnable(idx)

    yield

    for idx in state:
        glDisable(idx)


@contextlib.contextmanager
def configure_pixel_store(args):
    for name, value in args:
        glPixelStorei(name, value)

    yield

    for name, value in args:
        if name in [GL_PACK_ALIGNMENT, GL_UNPACK_ALIGNMENT]:
            glPixelStorei(name, 4)
        else:
            glPixelStorei(name, 0)


@contextlib.contextmanager
def bind_attributes(**attributes):
    program_handle = glGetIntegerv(GL_CURRENT_PROGRAM)

    vertices_count = 0
    bound = []
    for raw_name, values in attributes.items():
        # attribute key:  <attribute_name>(_(_[(o<attribute_offset>)|(d<divisor>)])+)?
        name = raw_name
        offset = 0
        divisor = None
        if '__' in raw_name:
            name, modifiers = name.split('__')[0], name.split('__')[1]
            for modifier in modifiers.split('_'):
                key, value = modifier[0], modifier[1:]
                if key == 'o':
                    offset = int(value)
                elif key == 'd':
                    divisor = int(value)
        location = glGetAttribLocation(program_handle, name)
        if location < 0:
            raise KeyError('Attribute not found: {} (from raw_name={})'.format(name, raw_name))

        location += offset
        if _np.isscalar(values):
            glVertexAttrib1f(location, values)
        elif isinstance(values, tuple):
            gl_vertex_attrib_f = [glVertexAttrib1f, glVertexAttrib2f, glVertexAttrib3f, glVertexAttrib4f][len(values)-1]
            gl_vertex_attrib_f(location, *values)
        else:
            if divisor is not None:
                glVertexAttribDivisor(location, divisor)
            else:
                vertices_count = max(vertices_count, _np.prod(values.shape[:-1]))

            glEnableVertexAttribArray(location)

            if values.dtype == _np.float64:
                values = values.astype(_np.float32)
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)  # [1, 2, ... n] -> [[1], [2], ... [n]]
            item_size = values.shape[-1]
            stride = values.strides[-2]
            if isinstance(values, _BufferObject):
                with values:
                    glVertexAttribPointer(location, item_size, _array_gl_type(values), GL_TRUE, stride, None)
            else:
                glVertexAttribPointer(location, item_size, _array_gl_type(values), GL_TRUE, stride, values)
            bound.append((location, divisor))

    yield vertices_count

    for location, divisor in bound:
        if divisor is not None:
            glVertexAttribDivisor(location, 0)
        glDisableVertexAttribArray(location)


def draw_attrib_arrays(primitive, indices=None, instance_n=None, **attributes):
    # attribute key:  <attribute_name>(_(_[(o<attribute_offset>)|(d<divisor>)])+)?
    with bind_attributes(**attributes) as vertices_count:
        def draw_indices(ptr):
            if instance_n is None:
                glDrawElements(primitive, indices.size, _array_gl_type(indices), ptr)
            else:
                glDrawElementsInstanced(primitive, indices.size, _array_gl_type(indices), ptr, instance_n)

        if isinstance(indices, ElementArrayBuffer):
            with indices:
                draw_indices(None)
        elif indices is not None:
            assert indices.min() >= 0
            assert indices.max() < vertices_count
            indices = indices.astype(_np.uint32)
            draw_indices(indices)
        else:
            if instance_n is None:
                glDrawArrays(primitive, 0,  vertices_count)
            else:
                glDrawArraysInstanced(primitive, 0, vertices_count, instance_n)


class Shader(_Bindable):

    def __init__(self, vert='', frag='', vert_version=None, frag_version=None):
        super().__init__()
        assert _gl_thread_local.IS_GL_THREAD
        self._handle = glCreateProgram()
        self.linked = False
        self.log = ""

        vert_version_prefix = "#version {}\n".format(vert_version) if vert_version is not None else ""
        vert = vert_version_prefix + "#define V2F out V2F\n" + vert
        frag_version_prefix = "#version {}\n".format(frag_version) if frag_version is not None else ""
        frag = frag_version_prefix + "#define V2F in V2F\n" + frag

        if not self.compile_shader([vert], GL_VERTEX_SHADER):
            logger.error('Source code of vertex shader:\n{}'.format(vert))
            raise Exception('Compilation of vertex shader failed!')
        if not self.compile_shader([frag], GL_FRAGMENT_SHADER):
            logger.error('Source code of fragment shader:\n{}'.format(frag))
            raise Exception('Compilation of fragment shader failed!')

        if not self.link():
            logger.error('Linking failed. Source code of vertex shader:\n{}'.format(vert))
            logger.error('Linking failed. Source code of fragment shader:\n{}'.format(frag))
            raise Exception()

        self.next_tex_slot = 0

    def compile_shader(self, code, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, code)
        glCompileShader(shader)
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        log = glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode('utf-8')
        self.log += log
        if status == 0:
            logger.error("Compiling failed, error:\n" + log)
            return False
        else:
            glAttachShader(self._handle, shader)
            return True

    def link(self):
        glLinkProgram(self._handle)
        status = glGetProgramiv(self._handle, GL_LINK_STATUS)
        log = glGetProgramInfoLog(self._handle)
        if isinstance(log, bytes):
            log = log.decode('utf-8')
        self.log += log
        if status == 0:
            logger.error("Linking failed, error:\n" + log)
            return False
        else:
            self.linked = True
            return True

    def bind(self):
        self.next_tex_slot = 0
        glUseProgram(self._handle)
        self._bound = True

    def unbind(self):
        glUseProgram(0)
        self._bound = False

    def uniform_tex(self, name, tex):
        assert self._bound
        slot = self.next_tex_slot
        glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(tex.target, tex)
        self.uniform_i(name, slot)
        glActiveTexture(GL_TEXTURE0)
        self.next_tex_slot += 1

    def uniform_i(self, name, values):
        loc = glGetUniformLocation(self._handle, name)
        if loc < 0:
            logger.warn('Uniform "{}" has location={}.'
                        ' (The reason maybe in name mismatch or simply compiler optimization)'
                        .format(name, loc))
            return
        values = _np.ascontiguousarray(_np.atleast_2d(values), _np.int32)
        n, m = values.shape
        functions = [glProgramUniform1iv, glProgramUniform2iv, glProgramUniform3iv, glProgramUniform4iv]
        gl_program_uniform_iv = functions[m - 1]
        gl_program_uniform_iv(self._handle, loc, n, values)

    def uniform_f(self, name, values):
        loc = glGetUniformLocation(self._handle, name)
        assert loc >= 0
        values = _np.ascontiguousarray(_np.atleast_2d(values), _np.float32)
        n, m = values.shape
        functions = [glProgramUniform1fv, glProgramUniform2fv, glProgramUniform3fv, glProgramUniform4fv]
        gl_program_uniform_fv = functions[m - 1]
        gl_program_uniform_fv(self._handle, loc, n, values)

    def uniform_matrix_f(self, name, mat, transpose=True):
        loc = glGetUniformLocation(self._handle, name)
        assert loc >= 0
        mat = _np.float32(mat)
        if mat.ndim < 3:
            mat = mat[_np.newaxis, ...]
        assert mat.shape[1] == mat.shape[2], 'Matrix is not square!'
        functions = [glUniformMatrix2fv, glUniformMatrix3fv, glUniformMatrix4fv]
        gl_uniform_matrix_nfv = functions[mat.shape[1] - 2]
        values = _np.ascontiguousarray(mat, _np.float32)
        gl_uniform_matrix_nfv(loc, len(mat), transpose, values)

    def locate_attribute(self, name):
        return glGetAttribLocation(self._handle, name)

    def bind_textures(self, **textures):
        for i, (name, texture) in enumerate(textures.items()):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(texture.target, texture)  # TODO: repair RAII!
            self.uniform_i(name, i)
        glActiveTexture(GL_TEXTURE0)


def create_image_texture(filepath):
    img = _np.array(image_open(filepath))
    assert img is not None
    h, w = img.shape[:2]
    texture = Texture2D()
    with texture:
        texture.set_params(LINEAR_LINEAR + CLAMP_TO_EDGE + NO_MIPMAPING + [(GL_TEXTURE_BORDER_COLOR, [0.0, 0.0, 0.0, 0.0])])
        with configure_pixel_store([(GL_UNPACK_ALIGNMENT, 1)]):
            glTexImage2D(texture.target, 0, GL_RGBA, w, h, 0, {3: GL_RGB, 4: GL_RGBA}[img.shape[2]], GL_UNSIGNED_BYTE, img[::-1])
    return texture
