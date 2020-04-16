from abc import ABCMeta, abstractmethod

import tensorflow as tf
from graphviz import Graph


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
    def create_module_layers(self, dtype, output_shape, output_activation) -> (tf.keras.layers.Layer, ...):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

    @abstractmethod
    def get_summary(self, output_shape, output_activation) -> str:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_summary()'")

    @abstractmethod
    def duplicate_parameters(self) -> list:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'duplicate_parameters()'")

    def visualize(self, view, save_dir_path):
        """"""
        filename = "viz_module_{}".format(self.module_id)
        if save_dir_path is not None and save_dir_path[-1] != '/':
            save_dir_path += '/'

        graph = Graph()
        graph.node("0", label=self.get_summary())

        graph.render(filename=filename, directory=save_dir_path, view=view, cleanup=True, format='svg')

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self) -> str:
        return self.merge_method

    def set_fitness(self, fitness):
        self.fitness = fitness
