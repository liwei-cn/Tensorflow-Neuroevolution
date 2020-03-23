from .base_population import BasePopulation
from ..encodings.base_genome import BaseGenome
from ..algorithms import CoDeepNEAT


class CoDeepNEATPopulation(BasePopulation):
    """"""

    def __init__(self, config):
        """"""
        # Initialize and register the CoDeepNEAT algorithm
        self.ne_algorithm = CoDeepNEAT(config)

        # Declare internal variables of the population
        self.environment = None
        self.bp_pop_size = 0
        self.mod_pop_size = 0
        self.generation_counter = None
        self.best_genome = None
        self.best_fitness = 0

        # Declare the actual containers for all blueprints and modules, which will later be initialized as dicts
        # associating the blueprint/module ids (dict key) to the respective blueprint or module (dict value). Also
        # declare the species objections, associating the species ids (dict key) to the id of the respective blueprint
        # or module (dict value)
        self.blueprints = None
        self.modules = None
        self.bp_species = None
        self.mod_species = None

    def initialize(self):
        """"""
        self.generation_counter = 0
        init_ret = self.ne_algorithm.initialize_population()
        self.blueprints, self.bp_species, self.bp_pop_size, self.modules, self.mod_species, self.mod_pop_size = init_ret

    def evolve(self):
        """"""
        self.generation_counter += 1
        evol_ret = self.ne_algorithm.evolve_population(self.blueprints, self.modules, self.bp_species, self.mod_species)
        self.blueprints, self.bp_species, self.bp_pop_size, self.modules, self.mod_species, self.mod_pop_size = evol_ret

    def evaluate(self):
        """"""
        best_genome, best_fitness = self.ne_algorithm.evaluate_population(environment=self.environment,
                                                                          blueprints=self.blueprints,
                                                                          modules=self.modules,
                                                                          mod_species=self.mod_species,
                                                                          generation=self.generation_counter,
                                                                          current_best_fitness=self.best_fitness)
        if best_genome is not None:
            self.best_genome = best_genome
            self.best_fitness = best_fitness

    def summary(self):
        """"""
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\n"
              "Generation: {}\n"
              "Blueprints population size: {}\n"
              "Modules population size: {}\n"
              "Best genome: {}\n"
              "Best genome fitness: {}\n"
              .format(self.generation_counter, self.bp_pop_size, self.mod_pop_size,
                      self.best_genome, self.best_fitness))

    def save_population(self, save_file_path):
        """"""
        pass

    def load_population(self, load_file_path):
        """"""
        pass

    def check_extinction(self) -> bool:
        """"""
        if len(self.blueprints) == 0 or len(self.modules) == 0:
            return True
        return False

    def set_environment(self, environment):
        """"""
        self.environment = environment
        input_shape = self.environment.get_input_shape()
        output_units = self.environment.get_output_units()
        self.ne_algorithm.set_input_output_shape(input_shape, output_units)

    def get_generation_counter(self) -> int:
        return self.generation_counter

    def get_best_genome(self) -> BaseGenome:
        """"""
        return self.best_genome


'''

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

    def get_best_genome(self) -> BaseGenome:
        """"""
        pass

    def get_worst_genome(self) -> BaseGenome:
        """"""
        pass

    def get_average_fitness(self) -> float:
        """"""
        pass
'''
