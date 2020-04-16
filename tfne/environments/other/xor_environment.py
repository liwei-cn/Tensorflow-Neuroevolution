import numpy as np
import tensorflow as tf

from ..base_environment import BaseEnvironment
from ...helper_functions import read_option_from_config


class XOREnvironment(BaseEnvironment):
    """"""

    def __init__(self, config, weight_training_eval):
        """"""
        # Register parameters
        self.weight_training_eval = weight_training_eval

        # Initialize corresponding input and output mappings
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize input and output shapes of the environment, which a neural network has to abide by
        self.input_shape = (2,)
        self.output_shape = (1,)

        # Initialize loss function to evaluate performance on either evaluation method
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # Initialize environment differently depending on if the genome's weights are first being trained before the
        # genome is assigned a fitness value
        if weight_training_eval:
            # Register the weight training variant as the genome eval function
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training

            # Initialize and read config supplied weight training parameters
            self.epochs = read_option_from_config(config, 'EVALUATION', 'evaluation_epochs')
            self.batch_size = read_option_from_config(config, 'EVALUATION', 'evaluation_batch_size')

            # Set standard parameters for weight training evaluation
            self.verbosity = 1
        else:
            # Register the NON weight training variant as the genome eval function
            self.eval_genome_fitness = self._eval_genome_fitness_non_weight_training

    def eval_genome_fitness(self, genome) -> float:
        """"""
        # TO BE OVERRIDEN
        pass

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        """"""
        # Get model and optimizer required for compilation
        model = genome.get_model()
        optimizer = genome.get_optimizer()
        if optimizer is None:
            raise RuntimeError("Genome to evaluate ({}) does not supply an optimizer and no standard optimizer defined"
                               "for XOR environment as of yet.")

        # Compile and train model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        model.fit(x=self.x, y=self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)

        # Evaluate and return its fitness
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))

        # FIXME BUG WORKAROUND
        # Sometimes loss function randomly returns NaN.
        # See bug report: https://github.com/tensorflow/tensorflow/issues/38457
        if tf.math.is_nan(evaluated_fitness):
            evaluated_fitness = 0.0

        return round(evaluated_fitness, 4)

    def _eval_genome_fitness_non_weight_training(self, genome) -> float:
        """"""
        # Evaluate and return its fitness by calling genome directly with input
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, genome(self.x))))
        return round(evaluated_fitness, 4)

    def replay_genome(self, genome):
        """"""
        print("Replaying Genome #{}:".format(genome.get_id()))
        print("Solution Values: \t{}\n".format(self.y))
        print("Predicted Values:\t{}\n".format(genome(self.x)))

    def is_weight_training(self) -> bool:
        """"""
        return self.weight_training_eval

    def set_verbosity(self, verbosity):
        """"""
        self.verbosity = verbosity

    def get_input_shape(self) -> (int,):
        return self.input_shape

    def get_output_shape(self) -> (int,):
        return self.output_shape
