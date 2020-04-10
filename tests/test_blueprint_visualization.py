import os
from datetime import datetime

import tensorflow as tf

from tfne.encodings.codeepneat.codeepneat_encoding import CoDeepNEATEncoding

encoding = CoDeepNEATEncoding(tf.float16)

blueprint_graph = dict()
gene_id, gene = encoding.create_blueprint_node(node=1, species=None)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=2, species=1)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=3, species=2)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=4, species=2)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=5, species=15)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=6, species=3)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=7, species=15)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_node(node=8, species=2)
blueprint_graph[gene_id] = gene

gene_id, gene = encoding.create_blueprint_conn(conn_start=1, conn_end=3)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=1, conn_end=4)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=3, conn_end=7)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=3, conn_end=2)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=3, conn_end=5)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=4, conn_end=2)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=5, conn_end=6)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=6, conn_end=2)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=7, conn_end=8)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=5, conn_end=8)
blueprint_graph[gene_id] = gene
gene_id, gene = encoding.create_blueprint_conn(conn_start=8, conn_end=2)
blueprint_graph[gene_id] = gene

# Create blueprint
blueprint_id, blueprint = encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                    output_shape=None,
                                                    output_activation=None,
                                                    optimizer_factory=None)
# Create directory for the visualizations
run_datetime_string = datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("run_%Y-%b-%d_%H-%M-%S/")
save_dir_path = os.path.abspath('./') + '/' + run_datetime_string
print(save_dir_path)
os.makedirs(save_dir_path)

blueprint.visualize(view=True, save_dir_path=save_dir_path)
