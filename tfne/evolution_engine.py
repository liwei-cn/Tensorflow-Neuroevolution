import ast

import ray

from .encodings.base_genome import BaseGenome


class EvolutionEngine:
    """"""

    def __init__(self,
                 population,
                 environment,
                 config,
                 num_cpus=None,
                 num_gpus=None,
                 max_generations=None,
                 max_fitness=None,
                 backup_agents=None):
        """"""
        # Register parameters
        self.population = population
        self.environment = environment
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        self.backup_agents = backup_agents

        # Read and process the evaluation config values relevant for the fitness evaluation of the population
        self.evaluation_epochs = ast.literal_eval(config['EVALUATION']['evaluation_epochs'])
        self.evaluation_batch_size = ast.literal_eval(config['EVALUATION']['evaluation_batch_size'])

        # Set up evaluation environment according to config and register environment with population
        self.environment.set_evaluation_parameters(epochs=self.evaluation_epochs, batch_size=self.evaluation_batch_size)
        self.population.set_environment(self.environment)

        # Initiate the Multiprocessing library ray
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

        # Create flag if backup agent supplied
        self.backup_agents_supplied = bool(len(self.backup_agents))

    def train(self) -> BaseGenome:
        """"""
        # Check if an uninitialized population has been supplied or if training commences on a pre-evolved population
        if self.population.get_generation_counter() is None:
            print("Initializing blank population...")
            self.population.initialize()
        else:
            print("Evolving a pre-evolved population")
        print("Initial state of the population:")
        self.population.evaluate()
        self.population.summary()

        # Do an initial run of the backup agents if supplied, saving the initial state of the population
        if self.backup_agents_supplied:
            for backup_agent in self.backup_agents:
                backup_agent(self.population)

        # Start possibly endless training loop
        while self._check_training_loop_exit_conditions():
            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate()

            # Give summary of population after each evaluation
            self.population.summary()

            # Call backup agents if supplied
            if self.backup_agents_supplied:
                for backup_agent in self.backup_agents:
                    backup_agent(self.population)

        # Determine best genome resulting from the evolution and then return it, after the process came to an end
        return self.population.get_best_genome()

    def _check_training_loop_exit_conditions(self) -> bool:
        if self.population.check_extinction():
            print("Population went extinct. Exiting evolutionary training loop...")
            return False
        if self.max_generations is not None and self.population.get_generation_counter() >= self.max_generations:
            print("Population reached specified maximum number of generations. Exiting evolutionary training loop...")
            return False
        if self.max_fitness is not None and self.population.get_best_genome().get_fitness() >= self.max_fitness:
            print("Population's best genome reached specified fitness threshold. Exiting evolutionary training loop...")
            return False
        return True
