import ast
import math

import numpy as np
import tensorflow as tf
from absl import logging

from ..base_environment import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """"""

    def __init__(self, config, weight_training_eval):
        """"""
        # Initialize corresponding input and output mappings
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize input and output shapes of the environment, which a neural network has to abide by
        self.input_shape = (2,)
        self.output_shape = (1,)

        # Initialize environment differently depending on if the genome's weights are first being trained before the
        # genome is assigned a fitness value
        self.weight_training_eval = weight_training_eval
        if weight_training_eval:
            # Register the weight training variant as the genome eval function
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training

            # Initialize appropriate Loss function and read config supplied weight training parameters
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
            self.epochs = ast.literal_eval(config['EVALUATION']['evaluation_epochs'])
            self.batch_size = ast.literal_eval(config['EVALUATION']['evaluation_batch_size'])
            logging.info("Config value for 'EVALUATION/evaluation_epochs': {}".format(self.epochs))
            logging.info("Config value for 'EVALUATION/evaluation_batch_size': {}".format(self.batch_size))
        else:
            # Register the NON weight training variant as the genome eval function
            self.eval_genome_fitness = self._eval_genome_fitness_non_weight_training

    def eval_genome_fitness(self, genome) -> float:
        """"""
        pass

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        """"""
        raise NotImplementedError()
        '''
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

        import random
        return random.random()
        '''

    def _eval_genome_fitness_non_weight_training(self, genome) -> float:
        """"""
        raise NotImplementedError()

    def replay_genome(self, genome):
        """"""
        raise NotImplementedError()
        '''
        model = genome.get_model()
        print("Replaying Genome {}...".format(genome.get_id()))
        print("Solution Values:\n{}".format(self.y))
        print("Predicted Values:\n{}".format(model.predict(self.x)))
        '''

    def is_weight_training(self) -> bool:
        return self.weight_training_eval

    def get_input_shape(self) -> tuple:
        return self.input_shape

    def get_output_shape(self) -> tuple:
        return self.output_shape
