import tensorflow as tf
from absl import logging

from .codeepneat_model import CoDeepNEATModel


class CoDeepNEATGenome:
    """"""

    def __init__(self,
                 genome_id,
                 origin_generation,
                 blueprint,
                 bp_assigned_modules,
                 dtype):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.origin_generation = origin_generation
        self.dtype = dtype

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules

        # Initialize internal variables
        self.fitness = None

        # Create optimizer and model
        self.configured_optimizer = self.blueprint.get_configured_optimizer()
        self.model = CoDeepNEATModel(blueprint=self.blueprint,
                                     bp_assigned_modules=self.bp_assigned_modules,
                                     dtype=self.dtype)

    def __str__(self) -> str:
        """"""
        logging.debug("\t\tToDo: Create proper str representation of genome")
        return "genome {} from gen {} with fitness {}".format(self.genome_id, self.origin_generation, self.fitness)

    def visualize(self, view, save_dir):
        """"""
        logging.warning("TODO: Implement codeepneat_genome.visualize()")

    def save_genotype(self, save_dir):
        """"""
        logging.warning("TODO: Implement codeepneat_genome.save_genotype()")

    def save_model(self, save_dir):
        """"""
        logging.warning("TODO: Implement codeepneat_genome.save_model()")

    def get_model(self) -> tf.keras.Model:
        """"""
        return self.model

    def get_configured_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.configured_optimizer

    def get_genotype(self) -> tuple:
        """"""
        return self.blueprint, self.bp_assigned_modules

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
