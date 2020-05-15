from __future__ import annotations

import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase


class CoDeepNEATModuleDenseDropout(CoDeepNEATModuleBase):
    """"""

    def __init__(self,
                 module_id,
                 merge_method,
                 units,
                 activation,
                 kernel_init,
                 bias_init,
                 dropout_rate):
        """"""
        # Register parameters
        super().__init__(module_id, merge_method)
        self.units = units
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dropout_flag = dropout_rate is not None
        self.dropout_rate = dropout_rate

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT DENSE Module | ID: {:>6} | Fitness: {:>6} | Units: {:>4} | Activ: {:>6} | Dropout: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.units,
                    self.activation,
                    "None" if self.dropout_rate is None else self.dropout_rate)

    def create_module_layers(self, dtype) -> (tf.keras.layers.Layer, ...):
        """"""
        # Create the basic keras Dense layer, needed in both variants of the Dense Module
        dense_layer = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=dtype)
        # Determine if Dense Module also includes a dropout layer, then return appropriate layer tuple
        if self.dropout_flag:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate, dtype=dtype)
            return dense_layer, dropout_layer
        else:
            return (dense_layer,)

    def create_mutation(self,
                        offspring_id,
                        config_params,
                        max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        pass
        return None, None

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         config_params,
                         max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        raise NotImplementedError()
        return None, None
