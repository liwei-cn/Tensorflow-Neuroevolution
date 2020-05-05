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
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement '__str__()'")

    def create_module_layers(self, dtype, output_shape, output_activation) -> (tf.keras.layers.Layer, ...):
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

    def get_summary(self, output_shape=None, output_activation=None) -> str:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_summary()'")

    def duplicate_parameters(self) -> list:
        """"""
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'duplicate_parameters()'")
