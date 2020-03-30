from .codeepneat_module_base import CoDeepNEATModuleBase
from .codeepneat_module_dense_network import CoDeepNEATModuleDenseNetwork


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
        self.dropout_rate = dropout_rate

    def __str__(self) -> str:
        """"""
        pass

    def create_module(self,
                      dtype,
                      output_shape=None,
                      output_activation=None) -> CoDeepNEATModuleDenseNetwork:
        """"""
        if output_shape is None:
            network_units = self.units
        else:
            if len(output_shape) > 1:
                raise RuntimeError("Dense Module being created with a multi-dimensional output shape")
            network_units = output_shape[0]
        network_activation = self.activation if output_activation is None else output_activation
        return CoDeepNEATModuleDenseNetwork(units=network_units,
                                            activation=network_activation,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer,
                                            dropout_rate=self.dropout_rate,
                                            dtype=dtype)
