from os import WCONTINUED
from platform import node
from re import L
from types import MethodDescriptorType
from networkx.readwrite import edgelist
from numpy.core.numeric import allclose
import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import pickle
from tqdm import tqdm
from utils.model import GNN2, AttExplainer
from sklearn.metrics import roc_auc_score

from test import BA2Motif
from torch_geometric.utils import to_networkx

torch.manual_seed(12345)

dataset = BA2Motif(root='data/BA2')
data = dataset[0]
G = to_networkx(data, to_undirected=True)
g_list = []

from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=1, shuffle=False)

open_file = open('node_id_ba2motif', 'rb')
node_id = pickle.load(open_file)
open_file.close()

def nodeExplainer(loader, model, att):
    fid = 0
    inv_fid = 0.0
    default_fid = 0.0
    count = 0
    w_count = 0
    d_count = 0
    sparsity = 0.0
    y_pred = []
    y_true = []
    p_all=0.0
    num_edge = 0
    true_count = 0
    all_count = 0
    for step, data in enumerate(loader):
        logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)

        att_ori = att[step].tolist()
        att_ori.sort(reverse=True)
        att_cf = sorted(att_ori)
        p = 0.0
        threshold = 0.1
        subset = []
        for i in range(len(node_id[step])):
            if len(node_id[step][i]) > 2:
                subset.append(node_id[step][i])
                all_count += len(node_id[step][i])
        subset = list(set(tuple(x) for x in subset))
        for i in att_ori:
            p += i
            if p > threshold:
                cut_line = i
                break
        subgraph = []
        for i in range(len(att[step].tolist())):
            if att[step].tolist()[i] >= cut_line:
                subgraph.append(i)

        subgraph = list(set(subgraph))
        for i in subgraph:
            for j in subset:
                if j in subset:
                    true_count += 1
        count_s = 0
        change_node = {}
        for i in range(data.x.size()[0]):
            if i in subgraph:
                count_s += 1
            else:
                change_node[i] = i - count_s
        count_p = 0
        inv_change_node = {}

        mask = torch.ones(data.x.size()[0], dtype=bool)
        inv_mask = torch.zeros(data.x.size()[0], dtype=bool)

        for k in subgraph:
            mask[k] = False
        new_x = torch.clone(data.x)[mask, :]
        new_feature = torch.clone(data.x)
        inv_new_feature = torch.clone(data.x)
        for i in range(data.x.size()[0]):
            if i in subgraph:
                new_feature[i, :] = 0
            else:
                inv_new_feature[i, :] = 0
        delete_idx = []
        default_feature = torch.zeros_like(data.x)
        new_x_list = []

        for i in range(data.edge_index.size()[1]):
            if data.edge_index[0][i] in subgraph or data.edge_index[1][i] in subgraph:
                new_x_list.append(data.edge_index[0][i])
                new_x_list.append(data.edge_index[1][i])
                delete_idx.append(i)
        new_x_list = list(set(new_x_list).union(set(subgraph)))

        for i in new_x_list:
            inv_mask[i] = True

        inv_new_x = data.x.detach()[inv_mask, :]

        for i in range(data.x.size()[0]):
            if i not in new_x_list:
                count_p +=1
            else:
                inv_change_node[i] = i - count_p
        delete_idx = list(set(delete_idx))
        for i in range(data.x.size()[0]):
            if i in new_x_list:
                new_feature[i, :] = 0
            else:
                inv_new_feature[i, :] = 0
        l = len(delete_idx)

        edge_mask = torch.ones(data.edge_index.size()[1], dtype=bool)
        inv_edge_mask = torch.zeros(data.edge_index.size()[1], dtype=bool)
        default_edge_mask = torch.zeros(data.edge_index.size()[1], dtype=bool)
        for k in delete_idx:
            edge_mask[k] = False
            inv_edge_mask[k] = True
        new_edge_index = torch.clone(data.edge_index)[:, edge_mask]
        inv_new_edge_index = torch.clone(data.edge_index)[:, inv_edge_mask]
        default_edge_index = torch.clone(data.edge_index)[:, default_edge_mask]
        for i, x in enumerate(new_edge_index):
            for j, t in enumerate(x):
                if t.item() in list(change_node.keys()):
                    new_edge_index[i][j] = change_node[t.item()]
        for i, x in enumerate(inv_new_edge_index):
            for j, t in enumerate(x):
                if t.item() in list(inv_change_node.keys()):
                    inv_new_edge_index[i][j] = inv_change_node[t.item()]
        new_data = Data(x=new_feature, edge_index=new_edge_index, y=data.y)
        inv_new_data = Data(x=inv_new_x, edge_index=inv_new_edge_index, y=data.y)
        default_data = Data(x=default_feature, edge_index=default_edge_index, y=data.y)

        batch = torch.zeros(new_data.x.size()[0], dtype=torch.int64)
        inv_batch = torch.zeros(inv_new_data.x.size()[0], dtype=torch.int64)

        if step in correct_list:
            d_count += 1
            num_edge += inv_new_data.edge_index.size()[1]
            
            sparsity = sparsity + (1 - l/data.edge_index.size()[1])
            logit_explain, _, _ = model(new_data.x, new_data.edge_index, batch)
            inv_logit_explain, _, _ = model(inv_new_data.x, inv_new_data.edge_index, inv_batch)
            default_logit, _, _ = model(default_data.x, default_data.edge_index, batch)
            a = torch.argmax(logit, dim=1)
            b = logit.squeeze().softmax(dim=0)[a].item()
            p_all += b
            c = logit_explain.squeeze().softmax(dim=0)[a].item()
            d = inv_logit_explain.squeeze().softmax(dim=0)[a].item()
            e = default_logit.squeeze().softmax(dim=0)[a].item()
            y_pred.append(inv_logit_explain.squeeze()[1].item())
            y_true.append(data.y.item())
            diff = b - c
            inv_diff = b - d
            default_diff = b - e
            fid += diff
            inv_fid += inv_diff
            default_fid += default_diff
    auc = roc_auc_score(y_true, y_pred)
    fid = fid / d_count
    sparsity = sparsity / d_count
    inv_fid = inv_fid / d_count
    default_fid = default_fid / d_count
    return fid, inv_fid, default_fid, auc, sparsity


def motifExplainer(loader, model, att, node_id):
    fid = 0
    inv_fid = 0.0
    default_fid = 0.0
    count = 0
    w_count = 0
    d_count = 0
    sparsity = 0.0
    y_pred = []
    y_true = []
    num_edge = 0
    true_count = 0
    all_count = 0
    for step, data in enumerate(loader):
        logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)

        att_ori = att[step].tolist()
        att_ori.sort(reverse=True)

        p = 0.0
        threshold = 0.999
        # threshold = 0.9
        for i in att_ori:
            p += i
            if p > threshold:
                cut_line = i
                break
        subgraph = []
        inv_subgraph = []
        edge_list = []
        inv_edge_list = []
        for i in range(len(att[step].tolist())):
            if att[step].tolist()[i] >= cut_line:
                edge_list.append(node_id[step][i])
                if len(node_id[step][i]) == 2:
                    true_count += 0
                    all_count += 2
                else:
                    true_count += len(node_id[step][i])
                    all_count += len(node_id[step][i])
                for j in node_id[step][i]:
                    subgraph.append(j)
            else:
                inv_edge_list.append(node_id[step][i])
                for j in node_id[step][i]:
                    inv_subgraph.append(j)
        subgraph = list(set(subgraph))
        inv_subgraph = list(set(inv_subgraph))
        mask = torch.zeros(data.x.size()[0], dtype=bool)
        inv_mask = torch.zeros(data.x.size()[0], dtype=bool)
        for i in subgraph:
            mask[i] = True
        for i in inv_subgraph:
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
                if data.edge_index[0][i] in j and data.edge_index[1][i] in j:
                    edge_idx.append(i)
        edge_idx = list(set(edge_idx))
        inv_edge_idx = []
        for j in inv_edge_list:
            for i in range(data.edge_index.size()[1]):
                if data.edge_index[0][i] in j and data.edge_index[1][i] in j:
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
        batch = torch.zeros(new_data.x.size()[0], dtype=torch.int64)
        inv_batch = torch.zeros(inv_new_data.x.size()[0], dtype=torch.int64)
        l = len(edge_idx)

        if step in correct_list:
            d_count += 1
            
            sparsity = sparsity + (1 - l/data.edge_index.size()[1])
            num_edge += new_data.edge_index.size()[1]
            logit_explain, _, _ = model(new_data.x, new_data.edge_index, batch)
            

            a = torch.argmax(logit, dim=1)
            b = logit.squeeze().softmax(dim=0)[a].item()
            c = logit_explain.squeeze().softmax(dim=0)[a].item()
            

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
    auc = roc_auc_score(y_true, y_pred)
    
    fid = fid / d_count
    sparsity = sparsity / d_count
    inv_fid = inv_fid / w_count
    default_fid = default_fid / d_count
    return fid, inv_fid, default_fid, auc, sparsity


open_file = open('ba2motif_motif_attention_emb', 'rb')
att_emb = pickle.load(open_file)
open_file.close()
open_file = open('ba2motif_motif_correct_list', 'rb')
correct_list = pickle.load(open_file)
open_file.close()
open_file = open('node_id_ba2motif', 'rb')
node_id = pickle.load(open_file)
open_file.close()
open_file = open('ba2motif_motif_attention', 'rb')
att = pickle.load(open_file)
open_file.close()
model = GNN2(input_channels=10, hidden_channels=128, output_channels=2)
model.load_state_dict(torch.load('model/gcn_ba2motif'))
model.eval()
fid, inv_fid, default_fid, auc, sparsity = motifExplainer(loader, model, att, node_id)
print(f'Fid: {fid}. Inv_fid: {inv_fid}. Default_fid: {default_fid}. AUC: {auc}. Sparsity: {sparsity}.')