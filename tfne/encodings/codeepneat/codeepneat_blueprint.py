import networkx as nx
import tensorflow as tf
from absl import logging


class CoDeepNEATBlueprintNode:
    def __init__(self, gene_id, node, species):
        self.gene_id = gene_id
        self.node = node
        self.species = species


class CoDeepNEATBlueprintConn:
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
                 blueprint_genotype,
                 optimizer,
                 learning_rate,
                 momentum,
                 nesterov,
                 output_units,
                 output_activation):
        """"""
        # Register parameters
        self.blueprint_id = blueprint_id
        self.blueprint_genotype = blueprint_genotype
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.output_units = output_units
        self.output_activation = output_activation

        # Initialize internal variables
        self.fitness = None

        # Initialize graph specific internal variables
        self.species = set()
        self.node_species = dict()
        self.node_dependencies = dict()
        self.graph_topology = list()

        # Process genotype
        self._process_graph_genotype()

    def __str__(self) -> str:
        """"""
        pass

    def _process_graph_genotype(self):
        """"""
        # Create set of species (self.species, set), assignment of nodes to their species (self.node_species, dict) as
        # well as the assignment of nodes to the nodes they depend upon (self.node_dependencies, dict)
        for gene in self.blueprint_genotype.values():
            if isinstance(gene, CoDeepNEATBlueprintConn):
                if gene.conn_end in self.node_dependencies:
                    self.node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    self.node_dependencies[gene.conn_end] = {gene.conn_start}
            else:  # if isinstance(gene, CoDeepNEATBlueprintNode):
                self.node_species[gene.node] = gene.species
                self.species.add(gene.species)
        # Remove species assigned to Input node
        self.species.remove(0)

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

    def get_blueprint_genotype(self) -> dict:
        """"""
        return self.blueprint_genotype

    def get_species(self) -> set:
        """"""
        return self.species

    def get_node_species(self) -> dict:
        """"""
        return self.node_species

    def get_node_dependencies(self) -> dict:
        """"""
        return self.node_dependencies

    def get_graph_topology(self) -> list:
        """"""
        return self.graph_topology

    def get_configured_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        if self.optimizer == "SGD":
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                           momentum=self.momentum,
                                           nesterov=self.nesterov)
        else:
            raise NotImplementedError("Available Optimizer '{}' not yet implemented".format(self.optimizer))

    def get_learning_rate(self) -> float:
        """"""
        return self.learning_rate

    def get_momentum(self) -> float:
        """"""
        return self.momentum

    def get_nesterov(self) -> bool:
        """"""
        return self.nesterov

    def get_output_units(self) -> int:
        """"""
        return self.output_units

    def get_output_activation(self) -> str:
        """"""
        return self.output_activation

    def get_id(self) -> int:
        return self.blueprint_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
