import argparse
import pickle
from utils.data_loader import FileLoader
from utils.motif_generater import GenMotif
from test import BA2Motif
# from ba3motif import BA3Motif
from torch_geometric.utils import to_networkx

class GenData(object):
    def __init__(self, g_list, graph_labels):
        self.g_list = g_list
        self.graph_labels = graph_labels

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data', default='Mutagenicity', help='data folder name')
    parser.add_argument('-edge_weight', type=bool, default=False, help='If data have edge labels')
    args, _ = parser.parse_known_args()
    return args

args = get_args()
dataset = BA2Motif(root='data/BA2')
g_list = []
g_labels = []
for i in range(len(dataset)):
    G = to_networkx(dataset[i], to_undirected=True)
    g_list.append(G)
    g_labels.append(dataset[i].y)
data = GenData(g_list, g_labels)
graph = GenMotif(data)
open_file = open('node_id_ba2motif', 'wb')
pickle.dump(graph.node_id, open_file)
open_file.close()