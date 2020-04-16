import ray
import graphviz
from absl import logging

from .encodings.base_genome import BaseGenome


class EvolutionEngine:
    """"""

    def __init__(self,
                 ne_algorithm,
                 environment,
                 num_cpus=None,
                 num_gpus=None,
                 max_generations=None,
                 max_fitness=None,
                 backup_agents=None):
        """"""
        # Register parameters
        self.ne_algorithm = ne_algorithm
        self.environment = environment
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        self.backup_agents = backup_agents

        # Register the evaluation environment through which the genomes are evaluated at the NE algorithm and set
        # environment verbosity level according to logging level
        self.ne_algorithm.register_environment(self.environment)
        if not logging.level_debug():
            self.environment.set_verbosity(0)

        # Initiate the Multiprocessing library ray and the graph visualization library
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        print("Using graphviz system library v{}.{}.{}".format(*graphviz.version()))

        # Create flag if backup agent supplied
        self.backup_agents_supplied = bool(len(self.backup_agents))

    def train(self) -> BaseGenome:
        """"""
        # Initialize population. If pre-evolved population was supplied will this be used as the initial population.
        self.ne_algorithm.initialize_population()

        # Start possibly endless training loop, only exited if population goes extinct, the maximum number of
        # generations or the maximum fitness has been reached
        while True:
            # Evaluate and summarize population
            generation_counter, best_fitness = self.ne_algorithm.evaluate_population()
            self.ne_algorithm.summarize_population()

            # Call backup agents if supplied
            if self.backup_agents_supplied:
                for backup_agent in self.backup_agents:
                    backup_agent(generation_counter, self.ne_algorithm)

            # Exit training loop if maximum number of generations or maximum fitness has been reached
            if self.max_fitness is not None and best_fitness >= self.max_fitness:
                print("Population's best genome reached specified fitness threshold.\n"
                      "Exiting evolutionary training loop...")
                break
            if self.max_generations is not None and generation_counter >= self.max_generations:
                print("Population reached specified maximum number of generations.\n"
                      "Exiting evolutionary training loop...")
                break

            # Evolve population
            population_extinct = self.ne_algorithm.evolve_population()

            # Exit training loop if population went extinct
            if population_extinct:
                print("Population went extinct.\n"
                      "Exiting evolutionary training loop...")
                break

        # Determine best genome from evolutionary process and return it. This should return the best genome of the
        # evolutionary process, even if the population went extinct.
        return self.ne_algorithm.get_best_genome()
