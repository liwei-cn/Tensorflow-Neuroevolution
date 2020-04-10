from typing import Callable
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
    """"""

    def __init__(self, module_id, merge_method):
        self.module_id = module_id
        self.merge_method = merge_method
        self.fitness = 0

    @abstractmethod
    def create_module_layers(self, dtype, output_shape, output_activation) -> (tf.keras.layers.Layer, ...):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

    @abstractmethod
    def get_summary(self, dtype, output_shape, output_activation) -> str:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_summary()'")

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self) -> Callable:
        return self.merge_method

    def set_fitness(self, fitness):
        self.fitness = fitness
