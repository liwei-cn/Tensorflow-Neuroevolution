from abc import ABCMeta, abstractmethod


class BaseEnvironment(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def eval_genome_fitness(self, genome) -> float:
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'eval_genome_fitness()'")

    @abstractmethod
    def replay_genome(self, genome):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'replay_genome()'")

    @abstractmethod
    def is_weight_training(self) -> bool:
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'is_weight_training()'")

    @abstractmethod
    def set_verbosity(self, verbosity):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'set_verbosity()'")

    @abstractmethod
    def get_input_shape(self) -> (int, ...):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_input_shape()'")

    @abstractmethod
    def get_output_shape(self) -> (int, ...):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_output_shape()'")
