import networkx as nx
from tqdm import tqdm

class GenData(object):
    def __init__(self, g_list, node_labels, graph_labels):
        self.g_list = g_list
        self.node_labels = node_labels
        self.graph_labels = graph_labels


class FileLoader(object):
    def __init__(self, args):
        self.args = args

    def load_data(self):
        data = self.args.data
        with open('datasets/%s/A.txt' % (data), 'r') as f:
            edges = f.read().splitlines()

        edges = [tuple(map(int, e.replace(" ", "").split(","))) for e in edges] 
        print("edges", len(edges))

        with open('datasets/%s/graph_indicator.txt' % (data), 'r') as f:
            g = f.readlines()
        g = [int(i) for i in g]
        print("g", len(g))

        weights = []
        if self.args.edge_weight:
            with open('datasets/%s/edge_labels.txt' % (data), 'r') as f:
                w = f.readlines()
            weights = [int(i) for i in w]
            print("weights:",len(weights))

        with open('datasets/%s/graph_labels.txt' % (data), 'r') as f:
            l = f.readlines()
        graph_labels = [int(i) for i in l]
        print("labels:", len(graph_labels))

        with open('datasets/%s/node_labels.txt' % (data), 'r') as f:
            nl = f.readlines()
        node_labels = [int(i[-2]) for i in nl]
        print("nodes_labels", len(node_labels))

        G_edges = [] 
        G_weight = []

        if self.args.edge_weight:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = [] 
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])

                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
            G_weight = []
            for i in tqdm(range(len(graph_labels)), desc="Create weights", unit='graphs'):
                weight = []
                for w in range(len(weights)):
                    if g[edges[w][0]-1] == i + 1:
                        weight.append(weights[w])
                    elif g[edges[w][0]-1] == i + 2:
                        break
                G_weight.append(weight)
        else:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = []
                weight = []
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])
                        weight.append(1)
                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
                G_weight.append(weight)
        g_list = []
        for i in tqdm(range(len(G_edges)), desc="Create original graph", unit='graphs'):
            g_list.append(self.gen_graph(G_edges[i], G_weight[i]))

        return GenData(g_list, node_labels, graph_labels)

    def gen_graph(self, data, weights):
        edges = data
        weights = weights
        g1 = []
        for i in range(len(edges)):
            l = list(edges[i])
            l.append(weights[i])
            g1.append(tuple(l))

        g = nx.Graph()
        g.add_weighted_edges_from(g1)
        return g