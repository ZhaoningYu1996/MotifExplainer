import torch
from torch._C import GraphExecutorState
import torch.nn as nn
from torch import FloatTensor as FT
import pickle
import json
from torch.nn.modules.loss import CrossEntropyLoss
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
import time
def my_loss(output, target):
    loss = torch.mean((output - target)**2*100)
    return loss

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='TUDataset', name='Mutagenicity')
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

train_dataset = dataset[:3000]
test_dataset = dataset[3000:]
from torch_geometric.loader import DataLoader

batch_size = 1
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

open_file = open('all_embed_mutag', 'rb')
all_embed = pickle.load(open_file)
open_file.close()
open_file = open('all_final_batch_mutag', 'rb')
all_final_batch = pickle.load(open_file)
open_file.close()
def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == targets).item(), len(preds)

def train(model, model2, optimizer, explainModel, classifier, loss_function):
    open_file = open('node_id_mutagenicity', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    model.to(device)
    explainModel.to(device)
    classifier.to(device)
    collect = 0
    correct1 = 0
    count1 = 0
    correct2 = 0
    count2 = 0
    l = 0.0
    attention = []
    attention_test = []
    min_loss = 10.0
    explainModel.train()
    t = 0.5
    fid = 0
    inv_fid = 0
    d_count = 0
    w_count = 0
    avg = 0
    true_count = 0
    avg_p = 0.0
    correct_list_train = []
    for step, data in enumerate(train_loader):
        if data.y == 0:
            data.to(device)
            d_count += 1
            logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)

            a = torch.argmax(logit, dim=1)
            att, embed, g_emb = explainModel(all_embed, graph_embed, all_final_batch, step)
            out = classifier(embed)
            out1 = classifier(graph_embed)
            out2 = classifier(g_emb)
            
            result1 = out1.argmax(dim=1)
            result2 = out2.argmax(dim=1)

            if result1 == result2:
                correct1 += 1
            count1 += 1

            b = logit.squeeze().softmax(dim=0)[a]
            c = out2.squeeze().softmax(dim=0)[a]
            diff = b - c
            avg += c.mean()
            avg_p += b.mean()
            loss = loss_function(out2, result1)

            l += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            collect += 1
        
    if (l / collect) < min_loss:
        min_loss = (l / collect)
        
        torch.save(explainModel.state_dict(), 'model/gcn_mutagenicity_mean_explain')
        att_emb = explainModel.att_embedding.data.cpu()
        open_file = open('motif_attention_emb', 'wb')
        pickle.dump(att_emb, open_file)
        open_file.close()
    print(f'The final loss: {min_loss}. The train accuracy: {correct1 / count1}.')
    avg = avg / count1
    return min_loss, correct1 / count1, correct_list_train

def test(model, explainModel, classifier):
    open_file = open('node_id_mutagenicity', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    count = 0
    correct = 0
    true_data = 0
    att_list = []
    explainModel.eval()
    correct_list = []
    fid = 0
    d_count = 0
    y_pred = []
    y_true = []
    num_edge = 0
    inv_fid = 0.0
    w_count = 0
    sparsity = 0
    e_p = 0.0
    prob_count = 0.0
    avg = 0.0
    avg_p = 0.0
    attVocab = {}
    nodeID = {}
    with torch.no_grad():
        pre_edge = 1
        for step, data in enumerate(test_loader):
            # graph_id = step + 3000
            if data.y == 0:
                data.to(device)
                all_e_count = 0
                e_count = 0
                batch = torch.zeros(len(node_id[step]), dtype=torch.int64).to(device)
                logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)
                att, embed, g_emb = explainModel(all_embed, graph_embed, all_final_batch, step)
                att = att.squeeze()
                attVocab[step] = att.tolist()
                nodeID[step] = node_id[step]
                a = torch.argmax(logit, dim=1)
                b = logit.squeeze().softmax(dim=0)[a].item()
                avg_p += b
                p = 0.0
                threshold = 0.9
                cut_line = 0.0
                subgraph = []
                inv_subgraph = []
                edge_list = []
                inv_edge_list = []
                for i in range(len(att.tolist())):
                    if att.tolist()[i] > cut_line:
                        # subgraph.append(i)
                        edge_list.append(node_id[step][i])
                        for j in node_id[step][i]:
                            print(step, j, i, pre_edge)
                            subgraph.append(j-pre_edge)
                    else:
                        # print(node_id[graph_id][i], pre_edge)
                        inv_edge_list.append(node_id[step][i])
                        for j in node_id[step][i]:
                            print(step, j, i, pre_edge)
                            inv_subgraph.append(j-pre_edge)

                subgraph = list(set(subgraph))
                inv_subgraph = list(set(inv_subgraph))
                mask = torch.zeros(data.x.size()[0], dtype=bool)
                inv_mask = torch.zeros(data.x.size()[0], dtype=bool)
                for i in subgraph:
                    print(i, pre_edge)
                    mask[i] = True
                for i in inv_subgraph:
                    # print(i)
                    inv_mask[i] = True
                new_x = torch.clone(data.x)[mask, :]
                new_inv_x = torch.clone(data.x)[inv_mask, :]

                count_s = 0
                change_node = {}
                for i in range(data.x.size()[0]):
                    if i not in subgraph:
                        count_s += 1
                    else:
                        change_node[i] = i - count_s
                count_p = 0
                inv_change_node = {}
                for i in range(data.x.size()[0]):
                    if i not in inv_subgraph:
                        count_p += 1
                    else:
                        inv_change_node[i] = i - count_p

                edge_idx = []
                for j in edge_list:
                    for i in range(data.edge_index.size()[1]):
                        if data.edge_index[0][i]+pre_edge in j and data.edge_index[1][i]+pre_edge in j:
                            edge_idx.append(i)
                edge_idx = list(set(edge_idx))
                inv_edge_idx = []
                for j in inv_edge_list:
                    for i in range(data.edge_index.size()[1]):
                        if data.edge_index[0][i]+pre_edge in j and data.edge_index[1][i]+pre_edge in j:
                            inv_edge_idx.append(i)

                edge_mask = torch.zeros(data.edge_index.size()[1], dtype=bool)
                inv_edge_mask = torch.zeros(data.edge_index.size()[1], dtype=bool)
                for i in edge_idx:
                    edge_mask[i] = True
                for i in inv_edge_idx:
                    inv_edge_mask[i] = True
                new_edge_index = torch.clone(data.edge_index)[:, edge_mask]
                new_inv_edge_index = torch.clone(data.edge_index)[:, inv_edge_mask]

                for i, x in enumerate(new_edge_index):
                    for j, t in enumerate(x):
                        if t.item() in list(change_node.keys()):
                            new_edge_index[i][j] = change_node[t.item()]
                for i, x in enumerate(new_inv_edge_index):
                    for j, t in enumerate(x):
                        if t.item() in list(inv_change_node.keys()):
                            new_inv_edge_index[i][j] = inv_change_node[t.item()]


                new_data = Data(x=new_x, edge_index=new_edge_index, y=data.y)
                inv_new_data = Data(x=new_inv_x, edge_index=new_inv_edge_index, y=data.y)
                new_data.to(device)
                inv_new_data.to(device)
                new_batch = torch.zeros(new_data.x.size()[0], dtype=torch.int64).to(device)
                inv_batch = torch.zeros(inv_new_data.x.size()[0], dtype=torch.int64).to(device)
                l = len(edge_idx)

                if new_x.size()[0] > 0:
                    d_count += 1
                    
                    sparsity = sparsity + (1 - new_data.edge_index.size()[1]/data.edge_index.size()[1])
                    num_edge += new_data.edge_index.size()[1]
                    logit_explain, _, _ = model(new_data.x, new_data.edge_index, new_batch)

                    if logit_explain.argmax(dim=1) == a:
                        true_data += 1
                    
                    c = logit_explain.squeeze().softmax(dim=0)[a].item()
                    avg += c

                    y_pred.append(logit_explain.squeeze()[1].item())
                    y_true.append(data.y.item())
                    diff = b - c
                    
                    fid += diff
                    if inv_new_data.x.size()[0] > 0:
                        inv_logit_explain, _, _ = model(inv_new_data.x, inv_new_data.edge_index, inv_batch)
                        d = inv_logit_explain.squeeze().softmax(dim=0)[a].item()
                        inv_diff = b - d
                        inv_fid += inv_diff
                        w_count += 1
                pre_edge += data.x.size(0)
            
    with open("att_mutag.json", "w") as outfile:
        json.dump(attVocab, outfile)
    with open("node_id_muatg.json", "w") as outfile:
        json.dump(nodeID, outfile)
    
    fid = fid / d_count
    sparsity = sparsity / d_count
    inv_fid = inv_fid / w_count

    print(f'Accuracy: {true_data / d_count}.')
    print(f'The Fid is {fid}.')
    print(f'The Inv_Fid is {inv_fid}.')
    print(f'Sparsity: {sparsity}.')
    print(f"Total Fid is {fid-inv_fid}.")
    open_file = open('motif_attention', 'wb')
    pickle.dump(att_list, open_file)
    open_file.close()
    return fid, inv_fid

model = GNN2(input_channels=14, hidden_channels=64, output_channels=2).to(device)

model2 = GNN2(input_channels=14, hidden_channels=64, output_channels=2).to(device)
classifier = nn.Linear(64, 2).to(device)
classifier.load_state_dict(torch.load('model/mlp_mutagenicity_new'))
classifier.eval()
explainModel = NewAttExplainer(64).to(device)
optimizer = torch.optim.Adam(explainModel.parameters(), lr=0.01)
criterion = CrossEntropyLoss().to(device)
model.load_state_dict(torch.load('model/gcn_mutagenicity_new'))
model2.load_state_dict(torch.load('model/gcn_mutagenicity_new'))
model.eval()
model2.eval()

classifier.eval()
best_acc = 0.0
best_loss = 1000
for epoch in tqdm(range(3), desc='Epoch'):
    loss, acc, correct_list_train = train(model, model2, optimizer, explainModel, classifier, criterion)
    if loss <= best_loss:
        best_loss = loss

explainModelTest = NewAttExplainer(64).to(device)
explainModelTest.load_state_dict(torch.load('model/gcn_mutagenicity_mean_explain'))
test_acc, correct_list = test(model, explainModelTest, classifier)
open_file = open('motif_correct_list', 'wb')
pickle.dump(correct_list, open_file)
open_file.close()
