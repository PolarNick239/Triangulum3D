#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

from abc import abstractmethod, ABCMeta


class Renderable(metaclass=ABCMeta):

    def set_projector(self, projector):
        pass

    @abstractmethod
    def render(self, camera,
               *, edges_mode=False):
        raise NotImplementedError()
