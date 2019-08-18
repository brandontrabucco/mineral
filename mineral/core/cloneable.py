"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Cloneable(ABC):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self._clone_args = args
        self._clone_kwargs = kwargs

    def clone(
        self
    ):
        return self.__class__(*self._clone_args, **self._clone_kwargs)

    @abstractmethod
    def copy_to(
        self,
        clone
    ):
        return NotImplemented
