import ast
import random

import tensorflow as tf
from absl import logging

from .codeepneat_genome import CoDeepNEATGenome
from .codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from .modules import CoDeepNEATModuleBase, CoDeepNEATModuleDense
from ..base_encoding import BaseEncoding


class CoDeepNEATEncoding(BaseEncoding):
    """"""

    def __init__(self, dtype):
        """"""
        # Register parameters
        self.dtype = dtype

        # Initialize internal variables
        self.genome_id_counter = 0
        self.mod_id_counter = 0
        self.bp_id_counter = 0
        self.bp_gene_id_counter = 0

        # Initialize tracking object for blueprint genes enabling unique ids for new blueprint genes as well as
        # identical gene ids for the same blueprint gene
        self.bp_gene_to_bp_gene_id_mapping = dict()

        # Declare Input shape and output_units, which are fixed and will later be set by the CoDeepNEAT algorithm
        self.input_shape = None
        self.output_units = None

    def create_blueprint_node(self, node, species) -> (int, CoDeepNEATBlueprintNode):
        """"""
        gene_key = (node,)
        if gene_key in self.bp_gene_to_bp_gene_id_mapping:
            gene_id = self.bp_gene_to_bp_gene_id_mapping[gene_key]
        else:
            self.bp_gene_id_counter += 1
            self.bp_gene_to_bp_gene_id_mapping[gene_key] = self.bp_gene_id_counter
            gene_id = self.bp_gene_id_counter

        return gene_id, CoDeepNEATBlueprintNode(gene_id, node, species)

    def create_blueprint_conn(self, conn_start, conn_end) -> (int, CoDeepNEATBlueprintConn):
        """"""
        gene_key = (conn_start, conn_end)
        if gene_key in self.bp_gene_to_bp_gene_id_mapping:
            gene_id = self.bp_gene_to_bp_gene_id_mapping[gene_key]
        else:
            self.bp_gene_id_counter += 1
            self.bp_gene_to_bp_gene_id_mapping[gene_key] = self.bp_gene_id_counter
            gene_id = self.bp_gene_id_counter

        return gene_id, CoDeepNEATBlueprintConn(gene_id, conn_start, conn_end)

    def create_blueprint(self,
                         blueprint_genotype,
                         optimizer,
                         learning_rate,
                         momentum,
                         nesterov,
                         output_activation) -> (int, CoDeepNEATBlueprint):
        """"""
        self.bp_id_counter += 1
        return self.bp_id_counter, CoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                       blueprint_genotype=blueprint_genotype,
                                                       optimizer=optimizer,
                                                       learning_rate=learning_rate,
                                                       momentum=momentum,
                                                       nesterov=nesterov,
                                                       output_units=self.output_units,
                                                       output_activation=output_activation)

    def create_dense_module(self,
                            merge_method,
                            units,
                            activation,
                            kernel_initializer,
                            bias_initializer,
                            dropout_flag,
                            dropout_rate) -> (int, CoDeepNEATModuleDense):
        """"""
        self.mod_id_counter += 1

        return self.mod_id_counter, CoDeepNEATModuleDense(module_id=self.mod_id_counter,
                                                          merge_method=merge_method,
                                                          units=units,
                                                          activation=activation,
                                                          kernel_initializer=kernel_initializer,
                                                          bias_initializer=bias_initializer,
                                                          dropout_flag=dropout_flag,
                                                          dropout_rate=dropout_rate)

    def create_genome(self, blueprint, bp_assigned_modules, generation) -> (int, CoDeepNEATGenome):
        """"""
        self.genome_id_counter += 1
        # Genome genotype: blueprint, bp_assigned_modules
        return self.genome_id_counter, CoDeepNEATGenome(genome_id=self.genome_id_counter,
                                                        origin_generation=generation,
                                                        blueprint=blueprint,
                                                        bp_assigned_modules=bp_assigned_modules,
                                                        dtype=self.dtype)

    def set_input_output_shape(self, input_shape, output_units):
        """"""
        # Discard input shape for now as Tensorflow automatically detects it.
        self.output_units = output_units
