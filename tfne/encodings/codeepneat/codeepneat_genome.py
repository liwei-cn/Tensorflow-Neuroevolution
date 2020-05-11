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

        # Initialize the compute graph, which is processed in order to create the TF model through keras functional API.
        # Each element is 4-tuple consisting of node, node dependencies, their merge method and the actual TF layers.
        compute_graph = list()

        # Get preprocessed information from blueprint required for TF model creation
        node_species = self.blueprint.get_node_species()
        node_dependencies = self.blueprint.get_node_dependencies()
        graph_topology = self.blueprint.get_graph_topology()

        # Create the compute graph by skipping the first level of the graph topology and then processing all nodes in
        # each level
        for topology_level in graph_topology[1:]:
            for node in topology_level:
                # Determine the specific module of the current node and the required nodes the current node depends upon
                current_node_assigned_module = self.bp_assigned_modules[node_species[node]]
                current_node_dependencies = tuple(node_dependencies[node])

                # Determine if merge needed and if so, configure it
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

        # Create dict of the node results, with the input node having the input layer. This way each node and its
        # corresponding layer can later be created by the functional API
        node_outputs = {1: inputs}

        # Process compute graph by iterating over its topologically sorted list, ensuring that no node is processed out
        # of order. The results of each node are calculated and appended to node_outputs by checking the results of the
        # dependencies, possibly merging them and then applying them to the node network
        for node, current_node_dependencies, merge_method, node_layers in compute_graph:
            # Create merged input from input nodes
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

            # Register the final output of the module layers as the output of this particular node
            node_outputs[node] = node_out

        # Create the static output layers through keras deserialization
        deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in self.output_layers]
        # Pipe the results of the dynamic graph of modules through the output layers
        outputs = node_outputs[2]
        for out_layer in deserialized_output_layers:
            outputs = out_layer(outputs)

        # Create the complete keras Model through the functional API by identifying the inputs and output layers
        model = tf.keras.Model(inputs, outputs)
        return model

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
