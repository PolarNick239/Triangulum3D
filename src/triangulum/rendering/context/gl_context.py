#
# Copyright (c) 2015, Nikolay Polyarnyi
# All rights reserved.
#

from abc import abstractmethod, ABCMeta


class GLContext(metaclass=ABCMeta):

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass

    def __enter__(self):
        self.activate()

    def __exit__(self, type, value, traceback):
        self.deactivate()
