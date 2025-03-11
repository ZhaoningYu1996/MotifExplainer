import argparse
import pickle
from utils.data_loader import FileLoader
from utils.motif_generater import GenMotif

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data', default='NCI1', help='data folder name')
    parser.add_argument('-edge_weight', type=bool, default=False, help='If data have edge labels')
    args, _ = parser.parse_known_args()
    return args

args = get_args()
data = FileLoader(args).load_data()
graph = GenMotif(data)
open_file = open('node_id_nci1', 'wb')
pickle.dump(graph.node_id, open_file)
open_file.close()
print(graph.node_id[0], graph.node_id[1], graph.node_id[2])
print(len(graph.node_id))