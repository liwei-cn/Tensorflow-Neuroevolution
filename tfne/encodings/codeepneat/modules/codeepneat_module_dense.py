import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase


class CoDeepNEATModuleDense(CoDeepNEATModuleBase):
    """"""

    def __init__(self,
                 module_id,
                 merge_method,
                 units,
                 activation,
                 kernel_initializer,
                 bias_initializer,
                 dropout_rate):
        """"""
        # Register parameters
        super().__init__(module_id, merge_method)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dropout_flag = dropout_rate is not None
        self.dropout_rate = dropout_rate

    def __str__(self) -> str:
        """"""
        pass

    def create_module_layers(self,
                             dtype,
                             output_shape=None,
                             output_activation=None) -> (tf.keras.layers.Layer, ...):
        """"""
        # Determine if the create layers should have module deviating parameters in case of an output module
        if output_shape is None:
            network_units = self.units
        else:
            if len(output_shape) > 1:
                raise RuntimeError("Dense Module being created with a multi-dimensional output shape")
            network_units = output_shape[0]
        network_activation = self.activation if output_activation is None else output_activation

        # Create the basic keras Dense layer, needed in both variants of the Dense Module
        dense_layer = tf.keras.layers.Dense(units=network_units,
                                            activation=network_activation,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer,
                                            dtype=dtype)
        # Determine if Dense Module also includes a dropout layer, then return appropriate layer tuple
        if self.dropout_flag:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate, dtype=dtype)
            return (dense_layer, dropout_layer)
        else:
            return (dense_layer,)
