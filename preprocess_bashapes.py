import argparse
import pickle
import networkx as nx
import itertools
from networkx.readwrite import edgelist
from utils.data_loader import FileLoader
from utils.motif_generater import GenMotif
from test import BA2Motif
# from ba3motif import BA3Motif
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import BAShapes
from torch_geometric.transforms import NormalizeFeatures

class GenData(object):
    def __init__(self, g_list, node_labels):
        self.g_list = g_list
        self.node_labels = node_labels

open_file = open('bashape', 'rb')
dataset = pickle.load(open_file)
open_file.close()
data = dataset[0]

g_list = []
g_labels = []

node_label = data.y
edge_index = data.edge_index
edge_label = data.edge_label
node_label = data.y.tolist()
edge_list = []
for i in range(edge_index.size()[1]):
    edge_list.append((edge_index[0, i].item()+1, edge_index[1, i].item()+1, {'weight': edge_label[i].item()}))
G = nx.Graph()
G.add_edges_from(edge_list)
g_list.append(G)
data = GenData(g_list, node_label)
graph = GenMotif(data)
setnode = set(itertools.chain.from_iterable(graph.node_id[0]))
open_file = open('node_id_bashape', 'wb')
pickle.dump(graph.node_id[0], open_file)
open_file.close()