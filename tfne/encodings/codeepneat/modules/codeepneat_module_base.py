from __future__ import annotations

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
    """"""

    def __init__(self, module_id, parent_mutation, merge_method):
        self.module_id = module_id
        self.parent_mutation = parent_mutation
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
    def create_mutation(self,
                        offspring_id,
                        config_params,
                        max_degree_of_mutation) -> (int, CoDeepNEATModuleBase):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_mutation()'")

    @abstractmethod
    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         config_params,
                         max_degree_of_mutation) -> (int, CoDeepNEATModuleBase):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_crossover()'")

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self) -> str:
        return self.merge_method
