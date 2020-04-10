import tensorflow as tf
from absl import logging
from graphviz import Digraph


class CoDeepNEATBlueprintNode:
    """"""

    def __init__(self, gene_id, node, species):
        self.gene_id = gene_id
        self.node = node
        self.species = species


class CoDeepNEATBlueprintConn:
    """"""

    def __init__(self, gene_id, conn_start, conn_end):
        self.gene_id = gene_id
        self.conn_start = conn_start
        self.conn_end = conn_end
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled


class CoDeepNEATBlueprint:
    """"""

    def __init__(self,
                 blueprint_id,
                 blueprint_graph,
                 output_shape,
                 output_activation,
                 optimizer_factory):
        """"""
        # Register internal parameters
        self.blueprint_id = blueprint_id

        # Register graph related parameters of blueprint
        self.blueprint_graph = blueprint_graph
        self.output_shape = output_shape
        self.output_activation = output_activation

        # Register the optimizer factory
        self.optimizer_factory = optimizer_factory

        # Initialize internal variables
        self.fitness = None

        # Declare graph related internal variables
        # species: set of all species present in blueprint
        # node_species: mapping of each node to its corresponding species
        # node dependencies: mapping of nodes to the node upon which they depend upon
        # graph topology: list of sets of dependency levels, with the first set being the nodes that depend on nothing,
        #                 the second set being the nodes that depend on the first set, and so on
        self.species = set()
        self.node_species = dict()
        self.node_dependencies = dict()
        self.graph_topology = list()

        # Process graph to set graph related internal variables
        self._process_graph()

    def __str__(self) -> str:
        """"""
        logging.warning("TODO: Implement codeepneat_blueprint.__str__()")

    def _process_graph(self):
        """"""
        # Create set of species (self.species, set), assignment of nodes to their species (self.node_species, dict) as
        # well as the assignment of nodes to the nodes they depend upon (self.node_dependencies, dict)
        for gene in self.blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintConn):
                if gene.conn_end in self.node_dependencies:
                    self.node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    self.node_dependencies[gene.conn_end] = {gene.conn_start}
            else:  # if isinstance(gene, CoDeepNEATBlueprintNode):
                self.node_species[gene.node] = gene.species
                self.species.add(gene.species)
        # Remove the 'None' species assigned to Input node
        self.species.remove(None)

        # Topologically sort the graph and save into self.graph_topology as a list of sets of levels, with the first
        # set being the layer dependent on nothing and the following sets depending on the values of the preceding sets
        node_deps = self.node_dependencies.copy()
        node_deps[1] = set()  # Add Input node 1 to node dependencies as dependent on nothing
        while True:
            # find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            if not dependencyless:
                # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
                # occured
                if node_deps:
                    raise ValueError("Circular Dependency Error when sorting the topology of the Blueprint graph")
                # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
                # regularly
                break
            # Add dependencyless nodes of current generation to list
            self.graph_topology.append(dependencyless)

            # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
            # dependencies of other nodes in order to create next iteration
            for node in dependencyless:
                del node_deps[node]
            for node, dep in node_deps.items():
                node_deps[node] = dep - dependencyless

    def visualize(self, view, save_dir_path):
        """"""
        # Create filename and adjust save_dir_path if not ending in slash, indicating folder path
        filename = "graph_blueprint_{}".format(self.blueprint_id)
        if save_dir_path[-1] != '/':
            save_dir_path += '/'

        # Define label string, summarizing the Blueprint
        label_string = f"CoDeepNEAT Blueprint (ID: {self.blueprint_id})\l" \
                       f"fitness: {self.fitness}\l" \
                       f"output shape: {self.output_shape}\l" \
                       f"output activation: {self.output_activation}\l" \
                       f"optimizer factory: {self.optimizer_factory}\l"

        # Create graph and set direction of graph starting from bottom to top. Include label string at bottom
        graph = Digraph(graph_attr={'rankdir': 'BT', 'label': label_string})

        # Traverse the blueprint graph and add edges and nodes to the graph
        for bp_gene in self.blueprint_graph.values():
            try:
                graph.edge(str(bp_gene.conn_start), str(bp_gene.conn_end))
            except AttributeError:
                graph.node(str(bp_gene.node), label="node: {}\lspecies: {}\l".format(bp_gene.node, bp_gene.species))

        # Highlight Input and Output Nodes with subgraphs
        with graph.subgraph(name='cluster_input') as input_cluster:
            input_cluster.node('1')
            input_cluster.attr(label='Input', color='blue')
        with graph.subgraph(name='cluster_output') as output_cluster:
            output_cluster.node('2')
            output_cluster.attr(label='Output', color='grey')

        # Render, save and optionally display the graph
        graph.render(filename=filename, directory=save_dir_path, view=view, cleanup=True, format='svg')

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer_factory.create_optimizer()

    def get_blueprint_graph(self) -> {int: object}:
        """"""
        return self.blueprint_graph

    def get_output_shape(self) -> (int, ...):
        """"""
        return self.output_shape

    def get_output_activation(self) -> str:
        """"""
        return self.output_activation

    def get_species(self) -> {int, ...}:
        """"""
        return self.species

    def get_node_species(self) -> {int: int}:
        """"""
        return self.node_species

    def get_node_dependencies(self) -> {int: int}:
        """"""
        return self.node_dependencies

    def get_graph_topology(self) -> ({int, ...}, ...):
        """"""
        return self.graph_topology

    def get_id(self) -> int:
        return self.blueprint_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
