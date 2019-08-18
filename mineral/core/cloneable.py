"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Cloneable(ABC):

    def __init__(
        self,
        class_name,
        *args,
        **kwargs
    ):
        self._clone_class_name = class_name
        self._clone_args = args
        self._clone_kwargs = kwargs

    def clone(
        self
    ):
        return self._clone_class_name(*self._clone_args, **self._clone_kwargs)

    @abstractmethod
    def copy_to(
        self,
        clone
    ):
        return NotImplemented
