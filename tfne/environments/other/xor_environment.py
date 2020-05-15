import numpy as np
import tensorflow as tf

from ..base_environment import BaseEnvironment, BaseEnvironmentFactory


class XOREnvironment(BaseEnvironment):
    """"""

    def __init__(self, verbosity, weight_training, epochs=None, batch_size=None):
        """"""
        # Initialize corresponding input and output mappings
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize loss function to evaluate performance on either evaluation method
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # Set the verbosity level
        self.verbosity = verbosity

        # If environment is set to be weight training then set eval_genome_function accordingly and save the supplied
        # weight training parameters
        if weight_training:
            # Register the weight training variant as the genome eval function
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training

            # Register parameters for the weight training variant of the genome eval function
            self.epochs = epochs
            self.batch_size = batch_size
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

        '''
        # Compile and train model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        model.fit(x=self.x, y=self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)

        # Evaluate and return its fitness
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))

        # FIXME BUG WORKAROUND
        # Sometimes loss function randomly returns NaN. Problem confirmed by Tensorflow.
        # See bug report: https://github.com/tensorflow/tensorflow/issues/38457
        if tf.math.is_nan(evaluated_fitness):
            evaluated_fitness = 0.0
        '''
        # FIXME temporary for quick testing of dev version
        import random
        evaluated_fitness = random.uniform(0, 100)

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


class XOREnvironmentFactory(BaseEnvironmentFactory):
    """"""

    def create_environment(self, verbosity, weight_training, epochs=None, batch_size=None) -> XOREnvironment:
        """"""
        return XOREnvironment(verbosity, weight_training, epochs, batch_size)

    def get_env_input_shape(self) -> (int,):
        """"""
        return (2,)

    def get_env_output_shape(self) -> (int,):
        """"""
        return (1,)
