from .codeepneat_genome import CoDeepNEATGenome
from .codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from .modules import CoDeepNEATModuleDense
from ..base_encoding import BaseEncoding


class CoDeepNEATEncoding(BaseEncoding):
    """"""

    def __init__(self, dtype):
        """"""
        # Register parameters
        self.dtype = dtype

        # Initialize internal counter variables
        self.genome_id_counter = 0
        self.mod_id_counter = 0
        self.bp_id_counter = 0
        self.bp_gene_id_counter = 0

        # Initialize container that maps a blueprint gene to its assigned blueprint gene id. If a new blueprint gene is
        # created will this container allow to check if that gene has already been created before and has been assigned
        # a unique gene id before. If the blueprint gene has already been created before will the same gene id be used.
        self.gene_to_gene_id = dict()

        # Initialize a counter for nodes and a history container, assigning the tuple of connection start and end a
        # previously assigned node or new node if not yet present in history.
        self.node_counter = 2
        self.conn_split_history = dict()

    def get_node_for_split(self, conn_start, conn_end) -> int:
        """"""
        conn_key = (conn_start, conn_end)
        if conn_key not in self.conn_split_history:
            self.node_counter += 1
            self.conn_split_history[conn_key] = self.node_counter

        return self.conn_split_history[conn_key]

    def create_blueprint_node(self, node, species) -> (int, CoDeepNEATBlueprintNode):
        """"""
        gene_key = (node,)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintNode(bp_gene_id, node, species)

    def create_blueprint_conn(self, conn_start, conn_end) -> (int, CoDeepNEATBlueprintConn):
        """"""
        gene_key = (conn_start, conn_end)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintConn(bp_gene_id, conn_start, conn_end)

    def create_blueprint(self,
                         blueprint_graph,
                         output_shape,
                         output_activation,
                         optimizer_factory) -> (int, CoDeepNEATBlueprint):
        """"""
        self.bp_id_counter += 1

        return self.bp_id_counter, CoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                       blueprint_graph=blueprint_graph,
                                                       output_shape=output_shape,
                                                       output_activation=output_activation,
                                                       optimizer_factory=optimizer_factory)

    def create_dense_module(self,
                            merge_method,
                            units,
                            activation,
                            kernel_initializer,
                            bias_initializer,
                            dropout_rate) -> (int, CoDeepNEATModuleDense):
        """"""
        self.mod_id_counter += 1

        return self.mod_id_counter, CoDeepNEATModuleDense(module_id=self.mod_id_counter,
                                                          merge_method=merge_method,
                                                          units=units,
                                                          activation=activation,
                                                          kernel_initializer=kernel_initializer,
                                                          bias_initializer=bias_initializer,
                                                          dropout_rate=dropout_rate)

    def create_genome(self, blueprint, bp_assigned_modules, generation) -> (int, CoDeepNEATGenome):
        """"""
        self.genome_id_counter += 1
        # Genome genotype: blueprint, bp_assigned_modules (dict mapping species to specific module for that species)
        return self.genome_id_counter, CoDeepNEATGenome(genome_id=self.genome_id_counter,
                                                        blueprint=blueprint,
                                                        bp_assigned_modules=bp_assigned_modules,
                                                        dtype=self.dtype,
                                                        origin_generation=generation)
