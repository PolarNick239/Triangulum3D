#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

import numpy as np

from triangulum.rendering.entities.camera import Camera, CameraMode

color_fp = '''
    const vec4 projector_stripes_colours[{stripes_types_number}] = vec4[{stripes_types_number}](
{stripes_colours}
    );

    vec4 getProjectorColor(vec2 uv) {{
        int stripe_id = int(floor(uv.x * {stripes_number})) % {stripes_types_number};
        return projector_stripes_colours[stripe_id];
    }}
'''


class StripesProjector(Camera):

    def __init__(self, *,
                 stripes_number=63, stripes_colours=((1, 0, 0), (0, 1, 0)),
                 course=-135, pitch=360-30, distance=5, target=(0, 0, 0),
                 aspect=1.0, fov_h=45, near=0.1, far=100000.0, mode: CameraMode=CameraMode.perspective):
        super(StripesProjector, self).__init__(
            course=course, pitch=pitch, distance=distance, target=target,
            aspect=aspect, fov_h=fov_h, near=near, far=far, mode=mode)
        self._stripes_number = stripes_number
        if np.array(stripes_colours).shape[1] == 3:
            stripes_colours = np.hstack([stripes_colours, [[1]] * len(stripes_colours)])
        self._stripes_colours = stripes_colours

    def get_shader_code(self):
        return color_fp.format(stripes_types_number=len(self._stripes_colours),
                               stripes_number=self._stripes_number,
                               stripes_colours=',\n'.join(['        vec4({})'.format(', '.join(map(str, colour)))
                                                           for colour in self._stripes_colours]))
