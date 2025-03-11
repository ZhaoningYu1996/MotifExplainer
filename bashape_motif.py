import torch
from torch._C import GraphExecutorState
import torch.nn as nn
from torch import FloatTensor as FT
import pickle

from torch.nn.modules.loss import CrossEntropyLoss
from utils.model import GNN, AttExplainer, NodeAttExplainer, BashapeAttExplainer
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import random
import numpy as np
from test import BA2Motif
from torch_geometric.utils import to_networkx

open_file = open('bashape', 'rb')
dataset = pickle.load(open_file)
open_file.close()
data = dataset[0]
batch = torch.zeros(data.num_nodes, dtype=torch.int64)

open_file = open('node_id_bashape', 'rb')
node_id = pickle.load(open_file)
open_file.close()

model = GNN(input_channels=10, hidden_channels=64, output_channels=4)
model.load_state_dict(torch.load('model/gcn_bashape_new'))
model.eval()

all_motif = []
all_mask = []
count = []
logit, node_embed = model(data.x, data.edge_index, batch)
for i in tqdm(range(300, 700)):
    if i % 5 == 0:
        motif_embed = []
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        motif_nodes = []
        for j in node_id:
            if i+1 in j:
                count.append(node_id.index(j))
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                for k in j:
                    mask[k-1] = True
                mask[i] = False
                embed = node_embed[mask, :]
                # print('hh')
                embed = torch.mean(embed, dim=0)
                motif_embed.append(embed)
                motif_nodes += j
        
        motif_nodes = list(set(motif_nodes))

        sub = []
        for k in motif_nodes:
            for j in node_id:
                if k in j:
                    if node_id.index(j) not in count:
                        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                        for t in j:
                            mask[t-1] = True
                        mask[i] = False
                        embed = node_embed[mask, :]
                        embed = torch.mean(embed, dim=0)
                        motif_embed.append(embed)
                    sub += j
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        for j in range(i, i+5):
            mask[j] = True
        embed = node_embed[mask, :]
        embed[0] = embed[0]*5
        embed = torch.mean(embed, dim=0)
        motif_embed.append(embed)
        motif_embed = torch.stack(motif_embed)
        motif_nodes += sub
        motif_nodes = list(set(motif_nodes))
        motif_nodes = [x - 1 for x in motif_nodes]

        all_motif.append(motif_embed)

open_file = open('bashape_embed', 'wb')
pickle.dump(all_motif, open_file)
open_file.close()