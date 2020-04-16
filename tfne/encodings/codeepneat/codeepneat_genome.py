import tensorflow as tf
from absl import logging
from graphviz import Digraph

from .codeepneat_blueprint import CoDeepNEATBlueprint
from .codeepneat_model import CoDeepNEATModel
from ..base_genome import BaseGenome


class CoDeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 blueprint,
                 bp_assigned_modules,
                 dtype,
                 origin_generation):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.dtype = dtype
        self.origin_generation = origin_generation

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules

        # Initialize internal variables
        self.fitness = None

        # Create optimizer and model
        self.optimizer = self.blueprint.create_optimizer()
        self.model = CoDeepNEATModel(blueprint=self.blueprint,
                                     bp_assigned_modules=self.bp_assigned_modules,
                                     dtype=self.dtype)

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model.predict(inputs)

    def __str__(self) -> str:
        """"""
        return "CoDeepNEATGenome || ID: {:>4} || Fitness: {:>6} || Origin Generation: {:>4}" \
            .format(self.genome_id, self.fitness, self.origin_generation)

    def visualize(self, view, save_dir_path):
        """"""
        # Create filename and adjust save_dir_path if not ending in slash, indicating folder path
        filename = "graph_genome_{}".format(self.genome_id)
        if save_dir_path is not None and save_dir_path[-1] != '/':
            save_dir_path += '/'

        # Define label string, summarizing the Genome
        label_string = f"CoDeepNEAT Genome (ID: {self.genome_id})\l" \
                       f"fitness: {self.fitness}\l" \
                       f"origin_generation: {self.origin_generation}\l" \
                       f"dtype: {self.dtype}\l"

        # Create graph and set direction of graph starting from bottom to top. Include label string at bottom
        graph = Digraph(graph_attr={'rankdir': 'BT', 'label': label_string})

        # Get blueprint graph and the blueprints hyperparameters. Blueprint graph is then traversed and converted into
        # the visualization graph, considering the special input node (1) and output node (2) attributes
        bp_graph = self.blueprint.get_blueprint_graph()
        output_shape = self.blueprint.get_output_shape()
        output_activation = self.blueprint.get_output_activation()
        for bp_gene in bp_graph.values():
            try:
                graph.edge(str(bp_gene.conn_start), str(bp_gene.conn_end))
            except AttributeError:
                if bp_gene.node == 1:
                    graph.node('1', label='Input Node')
                elif bp_gene.node == 2:
                    mod_str = self.bp_assigned_modules[bp_gene.species].get_summary(output_shape=output_shape,
                                                                                    output_activation=output_activation)
                    graph.node('2', label=mod_str)
                else:
                    mod_str = self.bp_assigned_modules[bp_gene.species].get_summary()
                    graph.node(str(bp_gene.node), label=mod_str)

        # Highlight Input and Output Nodes with subgraphs
        with graph.subgraph(name='cluster_input') as input_cluster:
            input_cluster.node('1')
            input_cluster.attr(label='Input', color='blue')
        with graph.subgraph(name='cluster_output') as output_cluster:
            output_cluster.node('2')
            output_cluster.attr(label='Output', color='grey')

        # Render, save and optionally display the graph
        graph.render(filename=filename, directory=save_dir_path, view=view, cleanup=True, format='svg')

    def save_genotype(self, save_dir_path):
        """"""
        logging.warning("TODO: Implement codeepneat_genome.save_genotype()")

    def save_model(self, save_dir_path):
        """"""
        logging.warning("TODO: Implement codeepneat_genome.save_model()")

    def get_model(self) -> tf.keras.Model:
        """"""
        return self.model

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer

    def get_genotype(self) -> (CoDeepNEATBlueprint, {int: int}):
        """"""
        return self.blueprint, self.bp_assigned_modules

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
