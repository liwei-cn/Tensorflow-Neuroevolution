import tensorflow as tf
from absl import logging

from .codeepneat_blueprint import CoDeepNEATBlueprint
from .modules.codeepneat_module_base import CoDeepNEATModuleBase
from ..base_genome import BaseGenome
from ...helper_functions import deserialize_merge_method


class CoDeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 blueprint,
                 bp_assigned_modules,
                 output_layers,
                 input_shape,
                 dtype,
                 origin_generation):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.input_shape = input_shape
        self.dtype = dtype
        self.origin_generation = origin_generation

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules
        self.output_layers = output_layers

        # Initialize internal variables
        self.fitness = None

        # Create optimizer and model
        self.optimizer = self.blueprint.create_optimizer()
        self.model = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        """"""

        compute_graph = list()

        node_species = self.blueprint.get_node_species()
        node_dependencies = self.blueprint.get_node_dependencies()
        graph_topology = self.blueprint.get_graph_topology()

        for topology_level in graph_topology[1:]:
            for node in topology_level:
                current_node_assigned_module = self.bp_assigned_modules[node_species[node]]
                current_node_dependencies = tuple(node_dependencies[node])

                if len(current_node_dependencies) > 1:
                    merge_method = deserialize_merge_method(current_node_assigned_module.get_merge_method())
                else:
                    merge_method = None

                # Create actual module layers
                node_layers = current_node_assigned_module.create_module_layers(dtype=self.dtype)

                # Create Compute tuple that has to be processed for the current node and append it to the compute graph
                compute_tuple = (node, current_node_dependencies, merge_method, node_layers)
                compute_graph.append(compute_tuple)

        # Now create actual Tensorflow model through functional keras API by defining inputs, then walking through
        # the compute graph and closing graph by adding the output layers
        inputs = tf.keras.Input(shape=self.input_shape, dtype=self.dtype)

        node_outputs = {1: inputs}
        for node, current_node_dependencies, merge_method, node_layers in compute_graph:
            if merge_method is not None:
                node_input_list = [node_outputs[node_dep] for node_dep in current_node_dependencies]
                # FIXME POSSIBLY ADJUST THE STATIC AXIS=1 PARAMETER FOR OTHER MERGE METHODS THAN CONCAT
                node_input = merge_method(node_input_list, axis=1)
            else:
                node_input = node_outputs[current_node_dependencies[0]]

            # Pipe the input through the sequential module layers
            node_out = node_input
            for layer in node_layers:
                node_out = layer(node_out)

            # Register the final output of the module layers as the node output
            node_outputs[node] = node_out

        deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in self.output_layers]
        outputs = node_outputs[2]
        for out_layer in deserialized_output_layers:
            outputs = out_layer(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

    '''
        # Create compute graph as list of 5-tuples that have to be called in FIFO order. Each 5-tuple contains a current
        # node, the nodes the current node depends upon, a flag if a merge is required due to multiple dependencies, the
        # actual merge method and finally a tuple containing the sequential order of the layers of the node module.
        self.compute_graph = list()

        # Request preprocessed information from blueprint required for TF model creation
        node_species = blueprint.get_node_species()
        node_dependencies = blueprint.get_node_dependencies()
        graph_topology = blueprint.get_graph_topology()

        # In creating the compute graph skip the first level and the last level of the graph topology. First level as
        # this level doesn't have to be computed as it takes on the input values. Last level as it requires special
        # considerations being the output layer.
        for topology_level in graph_topology[1:-1]:
            for node in topology_level:
                # Determine the specific module of the current node and the required nodes the current node depends upon
                current_node_assigned_module = bp_assigned_modules[node_species[node]]
                current_node_dependencies = tuple(node_dependencies[node])

                # Determine if merge needed and if so, configure it
                merge_flag = len(current_node_dependencies) > 1
                if merge_flag:
                    merge_method = deserialize_merge_method(current_node_assigned_module.get_merge_method())
                else:
                    merge_method = None

                # Create actual module layers in sequential order
                node_layers = current_node_assigned_module.create_module_layers(dtype=dtype)

                # Create Compute tuple that has to be processed for the current node and append it to the compute graph
                compute_tuple = (node, current_node_dependencies, merge_flag, merge_method, node_layers)
                self.compute_graph.append(compute_tuple)

        # Create compute tuple for the last level of the topology, which requires due to specially configured output
        # shape and output activation a special consideration
        # Determine the specific module of the out node and the required nodes the out node depends upon. The out node
        # is per framework specification always node 2
        out_node_ass_module = bp_assigned_modules[node_species[2]]
        out_node_dependencies = tuple(node_dependencies[2])

        # Determine if merge needed and if so, configure it
        out_merge_flag = len(out_node_dependencies) > 1
        if out_merge_flag:
            out_merge_method = deserialize_merge_method(out_node_ass_module.get_merge_method())
        else:
            out_merge_method = None

        # Create actual output module layers in sequential order with special output shape and activation
        out_node_layers = out_node_ass_module.create_module_layers(dtype=dtype,
                                                                   output_shape=blueprint.get_output_shape(),
                                                                   output_activation=blueprint.get_output_activation())

        # Create compute tuple that has to be processed for the output node and append it to the compute graph
        out_compute_tuple = (2, out_node_dependencies, out_merge_flag, out_merge_method, out_node_layers)
        self.compute_graph.append(out_compute_tuple)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """"""
        # Create dict of the node results, with the input node having the model inputs as its node result
        node_outputs = {1: tf.cast(inputs, dtype=self.dtype)}

        # Process compute graph by iterating over its topologically sorted list, ensuring that no node is processed out
        # of order. The results of each node are calculated and appended to node_outputs by checking the results of the
        # dependencies, possibly merging them and then applying them to the node network
        for node, node_dependencies, merge_flag, merge_method, node_layers in self.compute_graph:
            # Create merged input from input nodes
            if merge_flag:
                node_input_list = [node_outputs[node_dep] for node_dep in node_dependencies]
                node_input = merge_method(node_input_list, axis=1)
            else:
                node_input = node_outputs[node_dependencies[0]]

            # Pipe the input through the sequential module layers
            node_out = node_input
            for layer in node_layers:
                node_out = layer(node_out)

            # Register the final output of the module layers as the module output
            node_outputs[node] = node_out

        # Return result of the output node as result of the model
        return node_outputs[2]
    '''

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model.predict(inputs)

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT Genome | ID: {:>6} | Fitness: {:>6} | Blueprint ID: {:>6} | Module Species: {} | " \
               "Optimizer: {:>6} | Origin Gen: {:>4}".format(self.genome_id,
                                                             self.fitness,
                                                             self.blueprint.get_id(),
                                                             self.blueprint.get_species(),
                                                             self.blueprint.optimizer_factory.__class__.__name__,
                                                             self.origin_generation)

    def save_genotype(self, save_dir_path):
        """"""
        logging.warning("CoDeepNEATGenome.save_genotype() NOT YET IMPLEMENTED")

    def save_model(self, save_dir_path):
        """"""
        logging.warning("CoDeepNEATGenome.save_model() NOT YET IMPLEMENTED")

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_genotype(self) -> (CoDeepNEATBlueprint, {int: CoDeepNEATModuleBase}):
        """"""
        return self.blueprint, self.bp_assigned_modules

    def get_model(self) -> tf.keras.Model:
        """"""
        return self.model

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness
