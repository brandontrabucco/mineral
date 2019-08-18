"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Saver(ABC):

    @abstractmethod
    def save(
        self,
        iteration
    ):
        return  NotImplemented

    @abstractmethod
    def load(
        self,
        iteration
    ):
        return  NotImplemented
