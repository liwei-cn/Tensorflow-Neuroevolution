from typing import Union
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseGenome(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def __call__(self, inputs) -> tf.Tensor:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement '__call__()'")

    def __str__(self) -> str:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement '__str__()'")

    def visualize(self, view, save_dir_path):
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'visualize()'")

    def save_genotype(self, save_dir):
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'save_genotype()'")

    def save_model(self, save_dir):
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'save_model()'")

    def get_model(self) -> tf.keras.Model:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_model()'")

    def get_optimizer(self) -> Union[None, tf.keras.optimizers.Optimizer]:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_optimizer()'")

    def get_genotype(self) -> object:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_genotype()'")

    def get_id(self) -> int:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_id()'")

    def get_fitness(self) -> float:
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_fitness()'")

    def set_fitness(self, fitness):
        """"""
        raise NotImplementedError("Subclass of BaseGenome does not implement 'set_fitness()'")
