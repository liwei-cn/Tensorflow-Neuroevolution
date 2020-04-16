from abc import ABCMeta, abstractmethod

from ..encodings.base_genome import BaseGenome


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def register_environment(self, environment):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'register_environment()'")

    @abstractmethod
    def get_best_genome(self) -> BaseGenome:
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'get_best_genome()'")

    @abstractmethod
    def get_generation_counter(self) -> int:
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                  "'get_generation_counter()'")

    @abstractmethod
    def initialize_population(self):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                  "'initialize_population()'")

    @abstractmethod
    def evaluate_population(self) -> (int, int):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evaluate_population()'")

    @abstractmethod
    def summarize_population(self):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'summarize_evaluation()'")

    @abstractmethod
    def evolve_population(self) -> bool:
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evolve_population()'")

    @abstractmethod
    def save_population(self, save_file_path):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'save_population()'")

    @abstractmethod
    def visualize_population(self, save_file_path):
        """"""
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'visualize_population()'")
