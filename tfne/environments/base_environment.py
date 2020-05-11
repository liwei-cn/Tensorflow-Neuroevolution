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


class BaseEnvironmentFactory(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def create_environment(self, verbosity, weight_training, **kwargs) -> BaseEnvironment:
        """"""
        raise NotImplementedError("Subclass of BaseEnvironmentFactory does not implement 'create_environment()'")

    @abstractmethod
    def get_env_input_shape(self) -> (int, ...):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_input_shape()'")

    @abstractmethod
    def get_env_output_shape(self) -> (int, ...):
        """"""
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_output_shape()'")
