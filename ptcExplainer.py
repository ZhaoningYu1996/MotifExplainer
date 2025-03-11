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
# import matplotlib.pyplot as plt
import time
def my_loss(output, target):
    loss = torch.mean((output - target)**2*100)
    return loss

# Import dataset
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='TUDataset', name='PTC_MR')
# Split dataset
seed = 1
import os
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# torch.use_deterministic_algorithms(True)
# dataset = dataset.shuffle()
def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
device = torch.device('cpu')

train_dataset = dataset[:300]
test_dataset = dataset[300:]
# dataset = BA2Motif(root='data/BA2')
# data = dataset[0]
# # print(data.y)
# G = to_networkx(data, to_undirected=True)
# g_list = []
# for i in range(len(dataset)):
#     G = to_networkx(dataset[i], to_undirected=True)
#     g_list.append(G)
# print(len(g_list))
# dataset = dataset.shuffle()

# train_dataset = dataset[:800]
# val_dataset = dataset[800:900]
# test_dataset = dataset[900:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

batch_size = 1
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

open_file = open('all_embed_ptc', 'rb')
all_embed = pickle.load(open_file)
open_file.close()
open_file = open('all_final_batch_ptc', 'rb')
all_final_batch = pickle.load(open_file)
open_file.close()
# print(all_embed)
# print(all_embed[0].size())
# print(all_embed[0][1, :])
# print(all_final_batch[1])
# print(len(all_embed))
# print(len(all_final_batch))


# def cfLoss(pred, t):
#     p1 = pred[0][0]
#     p2 = pred[0][1]
#     if p1 > p2:
#         result = t*p1
#     else:
#         result = p2
#     return result

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    # _, targets = torch.max(y, dim=1)
    return torch.sum(preds == targets).item(), len(preds)

def train(model, model2, optimizer, explainModel, classifier, loss_function):
    open_file = open('node_id_ptc_mr', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    # print(node_id[0])
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
    min_loss = 100.0
    max_avg = 0.0
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
        # print(data.batch.size())
        # if data.y == 0:
            data.to(device)
            d_count += 1
            logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)
            if logit.argmax(dim=1) == data.y:
                # if step == 67:
                #     print(graph_embed)
                #     print(graph_embed.size())
                #     print(data.batch)
                # if step==66:
                #     print(node_embed)
                #     print(node_embed.size())
                # print(node_embed.size())
                a = torch.argmax(logit, dim=1)
                # if logit.argmax(dim=1) != data.y:
                #     continue



                # if a == data.y:
                    # Batch for motif
                # batch = torch.zeros(len(node_id[step]), dtype=torch.int64)
                # new_batch = []
                # for i in range(batch_size):
                #     if step*batch_size+i >= len(dataset):
                #         break
                #     new_batch.append(torch.full((1, len(node_id[step*batch_size+i])), i).squeeze())
                # new_batch = torch.cat(new_batch, dim=-1).to(device)
                # print(new_batch.size())
                
                # Batch for node
                # batch = data.batch
                att, embed, g_emb = explainModel(all_embed, graph_embed, all_final_batch, step)
                # print(att)
                # print(g_emb.grad_fn)

                # if step == 0:
                #     print(graph_embed)
                #     print(embed)
                #     print(g_emb)
                # print(g_emb.size())
                # print('hh')
                # print(explainModel.att_embedding.data.cpu())
                # print(explainModel.w.data.cpu())

                # logit_explain = classifier(motif_emb)
                # a = torch.argmax(logit, dim=1)
                
                # if inv_new_data.x.size()[0] > 0:
                #     inv_logit_explain, _, _ = model(inv_new_data.x, inv_new_data.edge_index, inv_batch)
                #     d = inv_logit_explain.squeeze().softmax(dim=0)[a]
                #     inv_diff = b - d
                #     inv_fid += inv_diff
                #     w_count += 1
                out = classifier(embed)
                out1 = classifier(graph_embed)
                out2 = classifier(g_emb)
                # out3 = classifier(motif_emb)
                
                result1 = out1.argmax(dim=1)
                result2 = out2.argmax(dim=1)
                # result3 = out3.argmax(dim=1)
                # if result1 == result3:
                #     correct_list_train.append(step)
                # if result1 == result2:
                #     true_count += 1
                # print(out1.size())
                # print(out2.size())
                # print(step)
                # print(out2.size())
                # print(result1.size())
                # c, num = accuracy(out, data.y)
                if result1 == result2:
                    correct1 += 1
                count1 += 1

                b = logit.squeeze().softmax(dim=0)[a]
                c = out2.squeeze().softmax(dim=0)[a]
                # print('hh')
                # print(b)
                # print(c)
                diff = b - c
                avg += c.mean()
                avg_p += b.mean()

                # loss =  diff
                # loss = logit_explain[0][a]
                # loss = t*loss_function(out2, result1) + (1-t)*loss_function(out3, result1)
                # loss = loss_function(out3, result1) + loss_function(out2, result1)
                # loss = loss_function(out2, result1) + loss_function(out, result1)
                loss = loss_function(out2, result1)
                # loss = loss_function(out2, result1) - 10*loss_function(out, result1)
                # loss = my_loss(embed, graph_embed)
                

                # print(loss)

                l += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                collect += 1
    # if avg / count1 > max_avg:
    if (l / collect) < min_loss:
        min_loss = (l / collect)
        # max_avg = avg / count1
        
        torch.save(explainModel.state_dict(), 'model/gcn_ptc_mean_explain')
        att_emb = explainModel.att_embedding.data.cpu()
        print(att_emb)
        open_file = open('ptc_motif_attention_emb', 'wb')
        pickle.dump(att_emb, open_file)
        open_file.close()
    print(f'The final loss: {min_loss}. The train accuracy: {correct1 / count1}.')
    print(f'Count: {count1}')
    avg = avg / count1
    print(f'AVG: {avg}.')
    print(f'AVG P: {avg_p / count1}.')
    print(f'True count: {true_count}.')
    print(f'The number of correct training instances: {len(correct_list_train)}.')
    return min_loss, correct1 / count1, correct_list_train

def test(model, explainModel, classifier):
    open_file = open('node_id_ptc_mr', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    print(node_id[0])
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
        for step, data in enumerate(test_loader):
            # if data.y == 0:
                data.to(device)
                all_e_count = 0
                e_count = 0
                # Batch for motif
                batch = torch.zeros(len(node_id[step]), dtype=torch.int64).to(device)
                # Batch for node
                # batch = data.batch
                # print(data.x.size())
                logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)
                # att_id, g_emb = explainModel(node_id, node_embed, batch, step)
                att, embed, g_emb = explainModel(all_embed, graph_embed, all_final_batch, step)
                att = att[0]
                attVocab[step] = att.tolist()
                nodeID[step] = node_id[step]
                # if step==4325:
                #     print(att)
                #     print(node_id[step])
                #     print(data.x)
                #     print(data.edge_index)
                # print(att)
                # print('hh')
                # print(explainModel.att_embedding.data.cpu())

                # logit_explain = classifier(motif_emb)
                a = torch.argmax(logit, dim=1)
                if a == data.y:
                    b = logit.squeeze().softmax(dim=0)[a].item()
                    avg_p += b
                    # if a == data.y:
                        # print(att)
                        # after_mask = torch.zeros(att.size()[0], dtype=bool)
                        # for i in range(att.size()[0]):
                        #     if att[i] > 0.0:
                        #         after_mask[i] = True
                        #         if len(node_id[step][i]) == 2:
                        #             e_count += 2
                        #         else:
                        #             e_count += len(node_id[step][i]) * 2
                        # all_e_count += data.edge_index.size()[1]
                        # e_p += 1 - e_count / all_e_count
                        # # print(after_mask)
                        # after_embed = embed[after_mask, :]

                    # after_embed = torch.t(torch.t(embed) * att)
                    # after_embed = global_mean_pool(after_embed, batch)
                    # # after_embed = torch.mean(after_embed, dim=0)
                    # log = classifier(after_embed)
                    # prob = log.squeeze().softmax(dim=0)[a].item()
                    # b = logit.squeeze().softmax(dim=0)[a].item()
                    # re = b - prob
                    # prob_count += re


                    
                    # att_ori = att.tolist()
                    # att_ori.sort(reverse=True)
                    # new_logit = classifier(g_emb)
                    # new_out = torch.argmax(new_logit, dim=1)
                    # if new_out == a:
                    #     true_data += 1

                    # cut_line = att_ori[1]
                    # print(cut_line)
                    # print(att_ori)
                    p = 0.0
                    threshold = 0.1
                    # for i in att_ori:
                    #     p += i
                    #     if p > threshold:
                    #         cut_line = i
                    #         break
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

                    # Generate node features
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

                    # Generate edges
                    edge_idx = []
                    # for i in range(data.edge_index.size()[1]):
                    #     if data.edge_index[0][i] in subgraph and data.edge_index[1][i] in subgraph:
                    #         edge_idx.append(i)
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

                    # Change node id
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
                    # if step < 4337:
                    #     ori_data = to_networkx(data)
                    #     nx.draw_spring(ori_data, with_labels = True)
                    #     plt.show()
                    #     plt.savefig(f'figures/before_{step}.png')
                    #     plt.close()
                        # print('hh')
                        # print(att)
                        # print(node_id[step])
                        # datax = to_networkx(new_data)
                        # nx.draw_spring(datax, with_labels = True)
                        # plt.show()
                        # plt.savefig(f'figures/after_{step}.png')
                        # plt.close()
                    # if step == 25:
                    #     print(att)

                    if new_x.size()[0] > 0:
                        d_count += 1
                        
                        sparsity = sparsity + (1 - new_data.edge_index.size()[1]/data.edge_index.size()[1])
                        num_edge += new_data.edge_index.size()[1]
                        # print(new_batch)
                        logit_explain, _, _ = model(new_data.x, new_data.edge_index, new_batch)
                        

                        # a = torch.argmax(logit, dim=1)
                        if logit_explain.argmax(dim=1) == a:
                            true_data += 1
                        
                        c = logit_explain.squeeze().softmax(dim=0)[a].item()
                        avg += c
                        # if step == 1:
                        #     print(b)
                        #     print(c)
                        

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
                
                        













                        # out1 = classifier(graph_embed)
                        # out2 = classifier(g_emb)
                        # result1 = out1.argmax(dim=1)
                        # result2 = out2.argmax(dim=1)
                        # c, num = accuracy(out2, out1)
                        # correct += c
                        # count += num

                        # if a == torch.argmax(logit, dim=1):
                        #     d_count += 1
                        #     b = logit.squeeze().softmax(dim=0)[a]
                        #     c = out2.squeeze().softmax(dim=0)[a]
                        #     diff = b - c
                        #     fid += diff

                        # # att_list.append(att_id)
                        # # print(g_emb.size())
                        # out1 = classifier(graph_embed)
                        # out2 = classifier(g_emb)
                        # if out1.argmax(dim=1)==out2.argmax(dim=1):
                        #     if out1.argmax(dim=1)==data.y:
                        #         correct_list.append(step)
                        # c, num = accuracy(out2, out1)
                        # # print(num)
                        # correct += c
                        # count += num
    with open("att_ptc.json", "w") as outfile:
        json.dump(attVocab, outfile)
    with open("node_id_ptc.json", "w") as outfile:
        json.dump(nodeID, outfile)
    # auc = roc_auc_score(y_true, y_pred)
    print(d_count)
    print(fid)
    print(inv_fid)
    print(avg/d_count)
    print(avg_p/d_count)
    
    fid = fid / d_count
    sparsity = sparsity / d_count
    inv_fid = inv_fid / w_count
    print(d_count)
    print(fid)
    print(inv_fid)
    print(sparsity)
    print(true_data)
    print(f'Accuracy: {true_data / d_count}.')
    # print(f'AUC: {auc}.')
    # default_fid = default_fid / d_count
    print(f'The Fid is {fid}.')
    print(f'AFTER Fid is {prob_count / d_count}.')
    print(f'Sparsity: {e_p / d_count}.')
    open_file = open('ptc_motif_attention', 'wb')
    pickle.dump(att_list, open_file)
    open_file.close()
    return fid, inv_fid



model = GNN2(input_channels=18, hidden_channels=64, output_channels=2).to(device)

model2 = GNN2(input_channels=18, hidden_channels=64, output_channels=2).to(device)
classifier = nn.Linear(64, 2).to(device)
classifier.load_state_dict(torch.load('model/mlp_ptc'))
classifier.eval()
explainModel = NewAttExplainer(64).to(device)
# for p in explainModel.fcl.parameters():
#             p.requires_grad = False
# nodeExplainModel = NodeAttExplainer(128, 2)
for param_tensor in explainModel.state_dict():
    print(param_tensor, "\t", explainModel.state_dict()[param_tensor].size())
optimizer = torch.optim.Adam(explainModel.parameters(), lr=0.01)
criterion = CrossEntropyLoss().to(device)
model.load_state_dict(torch.load('model/gcn_ptc'))
model2.load_state_dict(torch.load('model/gcn_ptc'))
model.eval()
model2.eval()

classifier.eval()
best_acc = 0.0
best_loss = 1000

for epoch in tqdm(range(300), desc='Epoch'):
    # start = time.time()
    loss, acc, correct_list_train = train(model, model2, optimizer, explainModel, classifier, criterion)
    end = time.time()
    # print(f'Cost time: {end-start}')
    if loss <= best_loss:
        best_loss = loss

explainModelTest = NewAttExplainer(64).to(device)
explainModelTest.load_state_dict(torch.load('model/gcn_ptc_mean_explain'))
# start = time.time()
test_acc, correct_list = test(model, explainModelTest, classifier)
# end = time.time()
# print(f'Cost time: {end-start}')
open_file = open('ptc_motif_correct_list', 'wb')
pickle.dump(correct_list, open_file)
open_file.close()
# print(f'Best loss is {best_loss}, best acc is {test_acc}.')
    # print(explainModel.att_embedding.weight.data.cpu())
    # train_node(dataset, model, criterion, optimizer, nodeExplainModel)
    # train_att_pool(dataset, model, criterion, optimizer, nodeExplainModel)