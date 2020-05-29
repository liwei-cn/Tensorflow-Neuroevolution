import tensorflow as tf


def create_model(blueprint, bp_assigned_modules, output_layers, input_shape, dtype) -> tf.keras.Model:
    """"""

    # Get preprocessed information from blueprint required for TF model creation
    node_species = blueprint.get_node_species()
    node_dependencies = blueprint.get_node_dependencies()
    graph_topology = blueprint.get_graph_topology()

    # Create the actual Tensorflow model through the functional keras API, starting with the inputs object and saving
    # the output of each layer in a dict that associates it with the node and serves for a later reference in the
    # functional style of model creation.
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    node_outputs = {1: inputs}

    # Create the TF model iteratively through the keras functional API by creating the single layers in the order in
    # which they are called, which is made possible through the topological sorting of the graph. Traverse this
    # topological sorting of the graph (though skip the first level as it always contains the Input node), process the
    # nodes of the level and then create the TF layers.
    for topology_level in graph_topology[1:]:
        for node in topology_level:
            # Determine the specific module of the current node and the required nodes the current node depends upon
            current_node_module = bp_assigned_modules[node_species[node]]
            current_node_dependencies = tuple(node_dependencies[node])

            # Determine if the node has multiple inputs and requires a merge. If so, configure it accordingly and create
            # merge input for the current node
            if len(current_node_dependencies) > 1:
                # Get current merge method, add dtype to its config and deserialize it
                merge_method_config = current_node_module.get_merge_method()
                merge_method_config['config']['dtype'] = dtype
                merge_method = tf.keras.layers.deserialize(merge_method_config)

                # Create list of all node outputs that the current node dependes upon and merge them
                node_input_list = [node_outputs[node_dep] for node_dep in current_node_dependencies]
                node_input = merge_method(node_input_list)
            else:
                node_input = node_outputs[current_node_dependencies[0]]

            # Create the sequential layers of the module and pipe the just created input through this node/module
            node_layers = current_node_module.create_module_layers(dtype=dtype)
            node_out = node_input
            for layer in node_layers:
                node_out = layer(node_out)

            # Register the final output of the sequential module layers as the output of the current node
            node_outputs[node] = node_out

    # Create the static output layers set by config and Pipe the results of the dynamic graph of modules through them.
    # The dynamic graph always has the output node 2, which is therefore the input to the output layers.
    deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in output_layers]
    outputs = node_outputs[2]
    for out_layer in deserialized_output_layers:
        outputs = out_layer(outputs)

    # Create the complete keras Model through the functional API by identifying the inputs and output layers
    model = tf.keras.Model(inputs, outputs)
    return model
