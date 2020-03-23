from abc import ABCMeta, abstractmethod


class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def __init__(self, module_id, merge_method):
        self.module_id = module_id
        self.merge_method = merge_method
        self.fitness = 0

    @abstractmethod
    def __str__(self) -> str:
        """"""
        pass

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self):
        return self.merge_method

    def set_fitness(self, fitness):
        self.fitness = fitness
