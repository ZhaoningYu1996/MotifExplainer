import pickle
from tqdm import tqdm
import torch
from torch._C import GraphExecutorState
import torch.nn as nn
from torch import FloatTensor as FT
import pickle

from torch.nn.modules.loss import CrossEntropyLoss
from torch_geometric.nn import GraphConv
from utils.model import GNN2, AttExplainer, NodeAttExplainer, NewAttExplainer
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F
import random
import numpy as np
from test import BA2Motif
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import networkx as nx
# import matplotlib.pyplot as plt

# Import dataset
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='TUDataset', name='PROTEINS')
# Split dataset
seed = 1
import os
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
device = torch.device('cpu')

train_dataset = dataset[:800]
test_dataset = dataset[800:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

batch_size = 1
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)
open_file = open('node_id_proteins', 'rb')
node_id = pickle.load(open_file)
open_file.close()
model = GNN2(input_channels=3, hidden_channels=64, output_channels=2).to(device)
model.load_state_dict(torch.load('model/gcn_proteins'))
model.eval()
conv2 = GraphConv(64, 64).to(device)
conv2.load_state_dict(torch.load('model/conv2_proteins'))
conv2.eval()
conv3 = GraphConv(64, 64).to(device)
conv3.load_state_dict(torch.load('model/conv3_proteins'))
conv3.eval()
classifier = nn.Linear(64, 2).to(device)
classifier.load_state_dict(torch.load('model/mlp_proteins'))
classifier.eval()
num_graph = len(dataset)
all_embed = []
all_final_batch = []
for step, data in enumerate(tqdm(train_loader, desc='data')):
    _, conv1_embed, _ = model(data.x, data.edge_index, data.batch)
    batch = data.batch
    edge_index = data.edge_index
    n_emb = data.x
    embed_list = []
    final_batch = []
    
    if (step+1)*batch_size <= num_graph:
            pre_edge = 0
            for i in range(step*batch_size, (step+1)*batch_size):
                count_batch = 0
                count_x = 0
                for k in batch:
                    if k == i - step*batch_size:
                        count_x += 1
                for k in node_id[i]:
                    count_batch += 1
                    edge_mask = torch.zeros(edge_index.size()[1], dtype=bool)
                    mask = torch.zeros(n_emb.size()[0], dtype=bool)
                    for j in range(edge_index.size()[1]):
                        if (edge_index[0, j]-pre_edge) in k and (edge_index[1, j]-pre_edge) in k:
                            edge_mask[j] = True
                    for j in k:
                        mask[j+pre_edge] = True
                    count_s = 0
                    change_node = {}
                    for j in range(n_emb.size()[0]):
                        if j-pre_edge not in k:
                            count_s += 1
                        else:
                            change_node[j] = j - count_s
                    new_x = data.x[mask, :]
                    new_edge_index = edge_index[:, edge_mask]
                    for k, x in enumerate(new_edge_index):
                        for j, t in enumerate(x):
                            if t.item() in list(change_node.keys()):
                                new_edge_index[k][j] = change_node[t.item()]
                    new_data = Data(x=new_x, edge_index=new_edge_index, y=data.y)
                    new_batch = torch.zeros(new_data.x.size()[0], dtype=torch.int64)
                    # new_conv2 = conv2(new_x, new_edge_index)
                    # new_conv2 = new_conv2.relu()
                    # new_conv3 = conv3(new_conv2, new_edge_index)
                    # new_conv3 = new_conv3.relu()
                    # motif_embed = global_mean_pool(new_conv3, new_batch)
                    _, _, motif_embed = model(new_data.x, new_data.edge_index, new_batch)
                    # print('hh')
                    # print(motif_embed)
                    embed_list.append(motif_embed)
                pre_edge += count_x
                # print(final_batch)
                final_batch=torch.full([1, count_batch], i - step*batch_size, dtype=torch.int64)
                # print(final_batch)
                if step == 217 or step == 300:
                    # print(node_id[i])
                    # print(edge_index)
                    # print(data.x)
                    print(final_batch[0])
                
                # print(final_batch.size())

    else:
        pre_edge = 0
        for i in range(step*batch_size, num_graph):
            count_batch = 0
            count_x = 0
            for k in batch:
                if k == i - step*batch_size:
                    count_x += 1
            for k in node_id[i]:
                count_batch += 1
                edge_mask = torch.zeros(edge_index.size()[1], dtype=bool)
                mask = torch.zeros(n_emb.size()[0], dtype=bool)
                for j in range(edge_index.size()[1]):
                    if (edge_index[0, j]-pre_edge) in k and (edge_index[1, j]-pre_edge) in k:
                        edge_mask[j] = True
                    
                for j in k:
                    mask[j+pre_edge] = True
                count_s = 0
                change_node = {}
                for j in range(n_emb.size()[0]):
                    if j-pre_edge not in k:
                        count_s += 1
                    else:
                        change_node[j] = j - count_s
                new_x = data.x[mask, :]
                new_edge_index = edge_index[:, edge_mask]
                for k, x in enumerate(new_edge_index):
                    for j, t in enumerate(x):
                        if t.item() in list(change_node.keys()):
                            new_edge_index[k][j] = change_node[t.item()]

                new_data = Data(x=new_x, edge_index=new_edge_index, y=data.y)
                new_batch = torch.zeros(new_data.x.size()[0], dtype=torch.int64)
                # new_conv2 = conv2(new_x, new_edge_index)
                # new_conv2 = new_conv2.relu()
                # new_conv3 = conv3(new_conv2, new_edge_index)
                # new_conv3 = new_conv3.relu()
                # motif_embed = global_mean_pool(new_conv3, new_batch)
                _, _, motif_embed = model(new_data.x, new_data.edge_index, new_batch)
                embed_list.append(motif_embed)
            pre_edge += count_x
            final_batch = torch.full((1, count_batch), i - step*batch_size, dtype=torch.int64)
    embed = torch.cat(embed_list, dim=0)
    # final_batch = torch.tensor(final_batch, dtype=torch.int64)
    final_batch = final_batch.clone().detach().to(torch.int64)
    if final_batch.size()[1] == 0:
        print(node_id[i])
    all_embed.append(embed)
    all_final_batch.append(final_batch)
open_file = open('all_embed_proteins', 'wb')
pickle.dump(all_embed, open_file)
open_file.close()
open_file = open('all_final_batch_proteins', 'wb')
pickle.dump(all_final_batch, open_file)
open_file.close()