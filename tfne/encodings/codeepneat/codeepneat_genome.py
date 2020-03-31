import numpy as np
import tensorflow as tf
from absl import logging

from .codeepneat_model import CoDeepNEATModel
from ..base_genome import BaseGenome


class CoDeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 blueprint,
                 bp_assigned_modules,
                 dtype,
                 origin_generation):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.dtype = dtype
        self.origin_generation = origin_generation

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules

        # Initialize internal variables
        self.fitness = None

        # Create optimizer and model
        self.optimizer = self.blueprint.create_optimizer()
        self.model = CoDeepNEATModel(blueprint=self.blueprint,
                                     bp_assigned_modules=self.bp_assigned_modules,
                                     dtype=self.dtype)

    def __call__(self, inputs) -> np.ndarray:
        """"""
        return self.model.predict(inputs)

    def __str__(self) -> str:
        """"""
        return "CoDeepNEATGenome || ID: {:>4} || Fitness: {:>6} || Origin Generation: {:>4}" \
            .format(self.genome_id, self.fitness, self.origin_generation)

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

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer

    def get_genotype(self) -> tuple:
        """"""
        return self.blueprint, self.bp_assigned_modules

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
