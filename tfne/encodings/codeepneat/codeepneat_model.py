import numpy as np
import tensorflow as tf


class CoDeepNEATModel(tf.keras.Model):
    """"""

    def __init__(self, blueprint, bp_assigned_modules, dtype):
        """"""
        super().__init__(dtype=dtype)
        # For Debugging
        # self.run_eagerly = True

        # initialize the list of three tuples that have to be called in FIFO order for the correct sequence of the
        # graph. The first element of the three tuple is the represented node, the second is the required input from
        # which preceding node and the third is the module function.
        self.modules = list()
        custom_layers = list()

        # Overwrite layers list of keras model
        self._layers = list()

        # Preprocess and create the TF model below
        # Call required preprocessed information from blueprint
        node_species = blueprint.get_node_species()
        bp_node_dependencies = blueprint.get_node_dependencies()
        bp_graph_topology = blueprint.get_graph_topology()

        # Skip the first level as this is only the input node as well as the last level in the automatic creation as
        # there are special considerations for this last level
        for topology_level in bp_graph_topology[1:-1]:
            for node in topology_level:
                assigned_module = bp_assigned_modules[node_species[node]]
                node_dependencies = tuple(bp_node_dependencies[node])
                merge_flag = len(node_dependencies) > 1
                if merge_flag:
                    merge_method = assigned_module.get_merge_method()
                else:
                    merge_method = None
                module_network = assigned_module.create_module(dtype=dtype)

                custom_layers = custom_layers + module_network.get_layers()

                module_tuple = (node, node_dependencies, merge_flag, merge_method, module_network)
                self.modules.append(module_tuple)

        # Append output node with a module following the specified output units and output activation
        out_assigned_module = bp_assigned_modules[node_species[2]]
        out_node_dependencies = tuple(bp_node_dependencies[2])
        out_merge_flag = len(out_node_dependencies) > 1
        if out_merge_flag:
            out_merge_method = out_assigned_module.get_merge_method()
        else:
            out_merge_method = None
        out_module_network = out_assigned_module.create_module(dtype=dtype,
                                                               output_units=blueprint.get_output_units(),
                                                               output_activation=blueprint.get_output_activation())
        custom_layers = custom_layers + out_module_network.get_layers()

        out_module_tuple = (2, out_node_dependencies, out_merge_flag, out_merge_method, out_module_network)
        self.modules.append(out_module_tuple)

        # Force overwrite private layers of keras model to register the right layers
        self._layers = custom_layers

        '''
        # TODO temp
        self.temp_layer = tf.keras.layers.Dense(units=1, activation='tanh')
        self.temp_dropout = tf.keras.layers.Dropout(0.4)
        '''

    def call(self, inputs, **kwargs) -> np.ndarray:
        """"""
        '''
        tf.print(inputs)
        tf.print(inputs.shape)
        tf.print(type(inputs))
        out = self.temp_layer(inputs)
        out = self.temp_dropout(out)

        tf.print(out)
        tf.print(out.shape)
        tf.print(type(out))
        return out
        '''

        '''
        out = self.temp_layer(inputs)
        out = self.temp_dropout(out)
        return out
        '''

        # FIXME WARNING, EVERYTHING BELOW THIS LINE IS UNTESTED DRAFT
        # Create dict of node results, with the input node having the model inputs as result
        node_outputs = {1: inputs}

        # TODO COMMENT
        for node, node_dependencies, merge_flag, merge_method, module_network in self.modules:
            if merge_flag:
                node_input_list = [node_outputs[node_dep] for node_dep in node_dependencies]
                node_input = merge_method(node_input_list, axis=1)
            else:
                node_input = node_outputs[node_dependencies[0]]

            node_output = module_network(node_input)
            node_outputs[node] = node_output

        # Return result of output node
        return node_outputs[2]

        '''
        # Network code of how it should result.
        layer_1_out = self.layer_1(inputs)
        layer_3_out = self.layer_3(layer_1_out)
        layer_9_out = self.layer_9(layer_3_out)
        layer_24_out = self.layer_24(layer_3_out)
        layer_11_out = self.layer_11(layer_1_out, layer_3_out) # ORDER IMPORTANT
        layer_5_out = self.layer_5(layer_24_out)
        layer_4_out = self.layer_4(layer_11_out, layer_24_out)
        layer_17_out = self.layer_17(layer_11_out)
        layer_10_out = self.layer_10(layer_4_out, layer_17_out)
        layer_8_out = self.layer_8(layer_5_out, layer_10_out, layer_17_out)
        layer_2_out = self.layer_2(layer_8_out, layer_9_out)

        return layer_2_out
        '''
