from .codeepneat_genome import CoDeepNEATGenome
from .codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from .modules import CoDeepNEATModuleDenseDropout
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

    def create_dense_dropout_module(self,
                                    merge_method,
                                    units,
                                    activation,
                                    kernel_init,
                                    bias_init,
                                    dropout_rate) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        module_id = self.get_next_module_id()
        return self.mod_id_counter, CoDeepNEATModuleDenseDropout(module_id=module_id,
                                                                 merge_method=merge_method,
                                                                 units=units,
                                                                 activation=activation,
                                                                 kernel_init=kernel_init,
                                                                 bias_init=bias_init,
                                                                 dropout_rate=dropout_rate)

    def create_blueprint(self,
                         blueprint_graph,
                         optimizer_factory) -> (int, CoDeepNEATBlueprint):
        """"""
        self.bp_id_counter += 1
        return self.bp_id_counter, CoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                       blueprint_graph=blueprint_graph,
                                                       optimizer_factory=optimizer_factory)

    def create_genome(self,
                      blueprint,
                      bp_assigned_modules,
                      output_layers,
                      input_shape,
                      generation) -> (int, CoDeepNEATGenome):
        """"""

        self.genome_id_counter += 1
        # Genome genotype: (blueprint, bp_assigned_modules, output_layers)
        return self.genome_id_counter, CoDeepNEATGenome(genome_id=self.genome_id_counter,
                                                        blueprint=blueprint,
                                                        bp_assigned_modules=bp_assigned_modules,
                                                        output_layers=output_layers,
                                                        input_shape=input_shape,
                                                        dtype=self.dtype,
                                                        origin_generation=generation)

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

    def get_node_for_split(self, conn_start, conn_end) -> int:
        """"""
        conn_key = (conn_start, conn_end)
        if conn_key not in self.conn_split_history:
            self.node_counter += 1
            self.conn_split_history[conn_key] = self.node_counter

        return self.conn_split_history[conn_key]

    def get_next_module_id(self) -> int:
        """"""
        self.mod_id_counter += 1
        return self.mod_id_counter
