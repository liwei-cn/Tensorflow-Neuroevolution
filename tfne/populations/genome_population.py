from ..encodings.base_genome import BaseGenome


class GenomePopulation:
    """"""

    def __init__(self, config, ne_algorithm):
        """"""
        raise NotImplementedError("NOT USED. SEE CODEEPNEATPOP")

    def initialize(self, input_shape, num_output):
        """"""
        pass

    def evolve(self):
        """"""
        pass

    def evaluate(self, environment_name, genome_eval_function):
        """"""
        pass

    def summary(self):
        """"""
        pass

    def check_extinction(self) -> bool:
        """"""
        pass

    def add_genome(self, genome_id, genome):
        """"""
        pass

    def delete_genome(self, genome_id):
        """"""
        pass

    def get_genome_ids(self) -> []:
        """"""
        pass

    def get_genome(self, genome_id) -> BaseGenome:
        """"""
        pass

    def get_pop_size(self) -> int:
        """"""
        pass

    def get_generation_counter(self) -> int:
        """"""
        pass

    def get_best_genome(self) -> BaseGenome:
        """"""
        pass

    def get_worst_genome(self) -> BaseGenome:
        """"""
        pass

    def get_average_fitness(self) -> float:
        """"""
        pass

    def save_population(self, save_file_path):
        """"""
        pass

    def load_population(self, encoding, load_file_path):
        """"""
        pass
