import numpy as np
import tensorflow as tf


class CoDeepNEATModuleDenseNetwork:
    """"""

    def __init__(self, units, activation, kernel_initializer, bias_initializer, dropout_flag, dropout_rate, dtype):
        """"""
        # Initialize core elements of the module network
        self.dense = tf.keras.layers.Dense(units=units,
                                           activation=activation,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           dtype=dtype)
        self.dropout_flag = dropout_flag
        if self.dropout_flag:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, dtype=dtype)

    def __call__(self, inputs) -> np.ndarray:
        """"""
        outputs = self.dense(inputs)
        if self.dropout_flag:
            outputs = self.dropout(outputs)

        return outputs

    def get_layers(self) -> list:
        """"""
        if self.dropout_flag:
            return [self.dense, self.dropout]
        else:
            return [self.dense]
