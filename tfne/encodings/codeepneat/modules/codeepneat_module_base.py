from abc import ABCMeta, abstractmethod

import tensorflow as tf


class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
    """"""

    def __init__(self, module_id, merge_method):
        self.module_id = module_id
        self.merge_method = merge_method
        self.fitness = 0

    @abstractmethod
    def __str__(self) -> str:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement '__str__()'")

    @abstractmethod
    def create_module_layers(self, dtype) -> (tf.keras.layers.Layer, ...):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

    @abstractmethod
    def copy_parameters(self) -> tuple:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'copy_parameters()'")

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self) -> str:
        return self.merge_method
