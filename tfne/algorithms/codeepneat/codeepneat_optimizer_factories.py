import tensorflow as tf


class SGDFactory:
    """"""

    def __init__(self, learning_rate, momentum, nesterov):
        """"""
        # Register parameters for optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def __str__(self) -> str:
        """"""
        return "SGD Optimizer (lr: {}; mom: {}; nstv: {}".format(self.learning_rate, self.momentum, self.nesterov)

    def create_optimizer(self) -> tf.keras.optimizers.SGD:
        """"""
        return tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                       momentum=self.momentum,
                                       nesterov=self.nesterov)

    def get_parameters(self) -> (float, float, bool):
        """"""
        return self.learning_rate, self.momentum, self.nesterov
