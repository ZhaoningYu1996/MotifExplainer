import networkx as nx
import torch
from tqdm import tqdm
import math
import numpy as np
import copy
import gc

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

class GenMotif(object):
    def __init__(self, data):
        self.data = data
        # self.nodes_labels = data.node_labels # For real datasets
        self.vocab = {}
        self.node_id = []
        self.gen_components()
        # self.update_node_id() # For real datasets


    def gen_components(self):
        g_list = self.data.g_list
        h_g = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            n_id = []
            mcb = nx.cycle_basis(g_list[g])
            if g==500:
                print(mcb)
            mcb_tuple = [tuple(ele) for ele in mcb]

            edges = []
            for e in g_list[g].edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            # edges = list(set(edges))

            for e in edges:
                n_id.append(list(e))
                # Change with different datasets
                # weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                weight = 1
                # edge = ((self.nodes_labels[e[0]-1], self.nodes_labels[e[1]-1]), weight)
                edge = ((1, 1), weight)
                clique_id = self.add_to_vocab(edge)

            for m in mcb_tuple:
                n_id.append(list(m))
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = []
                for i in range(len(m)):
                    # ring.append(self.nodes_labels[m[i]-1])
                    ring.append(0.1)
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
            self.node_id.append(n_id)

    def update_node_id(self):
        for i in self.node_id:
            min_x = 1000000
            for j in i:
                for k in j:
                    if k < min_x:
                        min_x = k
            for j in i:
                for k in range(len(j)):
                    j[k] = j[k] - min_x



    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        for i in range(len(c)):
            if (c, weight) in self.vocab:
                return self.vocab[(c, weight)]
            else:
                c = self.shift_right(c)
                weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(list(self.vocab.keys()))
        return self.vocab[(c, weight)]

    @staticmethod
    def shift_right(l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            # weight = g.get_edge_data(ring[i], ring[i+1])['weight']
            weight = 1
            weight_list.append(weight)
        # weight = g.get_edge_data(ring[-1], ring[0])['weight']
        weight = 1
        weight_list.append(weight)
        return weight_list

