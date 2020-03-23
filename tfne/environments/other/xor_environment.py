import math

import numpy as np
import tensorflow as tf
from absl import logging

from ..base_environment import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """"""

    def __init__(self):
        """"""
        # Initialize correct input and output mappings
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize appropriate Loss function
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # Declare hyperparameters of the evaluation
        self.epochs = None
        self.batch_size = None

        # Initialize fixed input/output attributes of the specific environment, which are required in order for a neural
        # network to be applied to this environment
        self.input_shape = (2,)
        self.num_output = 1

    def eval_genome_fitness(self, genome) -> float:
        """"""
        # TODO REDO
        # Get model and optimizer required for compilation
        model = genome.get_model()
        optimizer = genome.get_configured_optimizer()

        # Compile and train model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        model.fit(x=self.x, y=self.y, epochs=self.epochs, batch_size=self.batch_size)

        # Evaluate and return its fitness
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))

        # TODO Resolve Workaround
        # Introduce temporary check for NaN as sometimes loss seems to yield NaN arbitrarily
        if math.isnan(evaluated_fitness):
            evaluated_fitness = float(0)

        return round(evaluated_fitness, 3)

        '''
        import random
        return random.random()
        '''

    def replay_genome(self, genome):
        """"""
        # TODO REDO
        model = genome.get_model()
        logging.info("Replaying Genome {}...".format(genome.get_id()))
        logging.info("Solution Values:\n{}".format(self.y))
        logging.info("Predicted Values:\n{}".format(model.predict(self.x)))

    def set_evaluation_parameters(self, epochs, batch_size):
        """"""
        self.epochs = epochs
        self.batch_size = batch_size

    def get_input_shape(self) -> ():
        return self.input_shape

    def get_output_units(self) -> int:
        return self.num_output
