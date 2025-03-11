import torch
from torch._C import GraphExecutorState
import torch.nn as nn
from torch import FloatTensor as FT
import pickle

from torch.nn.modules.loss import CrossEntropyLoss
from utils.model import GNN2, AttExplainer, NodeAttExplainer
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import random
import numpy as np
from test import BA2Motif
from torch_geometric.utils import to_networkx

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
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
dataset = BA2Motif(root='data/BA2')
data = dataset[0]
G = to_networkx(data, to_undirected=True)
g_list = []
dataset = dataset.shuffle()

train_dataset = dataset[:800]
val_dataset = dataset[800:900]
test_dataset = dataset[900:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

batch_size = 1
train_loader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)


def accuracy(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    _, targets = torch.max(y, dim=1)
    return torch.sum(preds == targets).item(), len(preds)

def train(model, model2, optimizer, explainModel, classifier, loss_function, min_loss):
    open_file = open('node_id_ba2motif', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    collect = 0
    correct1 = 0
    count1 = 0
    correct2 = 0
    count2 = 0
    l1 = 0.0
    l2 = 0.0
    attention = []
    attention_test = []
    # min_loss = 10.0
    explainModel.train()
    for step, data in enumerate(train_loader):
        logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)
        if logit.argmax(dim=1) != data.y:
            continue
        batch = torch.zeros(len(node_id[step]), dtype=torch.int64)
        _, _, _, g_emb, g_emb_inv = explainModel(node_id, node_embed, batch, step)
        out1 = classifier(graph_embed)
        out2 = classifier(g_emb)
        out3 = classifier(g_emb_inv)
        result1 = out1.argmax(dim=1)
        # result2 = out2.argmax(dim=1)
        inv_result = torch.tensor([0]) if result1.item() == 1 else torch.tensor([1])
        # print(result1, inv_result)
        # print(stop)
        c, num = accuracy(out2, out1)
        correct1 += c
        count1 += num
        # print(out2.size(), out3.size(), result1.size(), inv_result.size())
        loss = loss_function(out2, result1) + 0.01*loss_function(out3, inv_result)
        l1 += loss_function(out2, result1).item()
        l2 += 0.01*loss_function(out3, inv_result).item()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        collect += 2
    
    if ((l1+l2) / collect) < min_loss:
        print("Update model")
        min_loss = ((l1+l2) / collect)
        torch.save(explainModel.state_dict(), 'model/gcn_ba2motif_motif_explain')
        att_emb = explainModel.att_embedding.data.cpu()
        open_file = open('ba2motif_motif_attention_emb', 'wb')
        pickle.dump(att_emb, open_file)
        open_file.close()
    print(f'The loss 1: {l1 / collect}, the loss 2: {l2 / collect}. The train accuracy: {correct1 / count1}.')
    print(f'Count: {count1}')
    return min_loss, correct1 / count1

def test(model, explainModel, classifier):
    open_file = open('node_id_ba2motif', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    count = 0
    correct = 0
    att_list = []
    explainModel.eval()
    correct_list = []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            batch = torch.zeros(len(node_id[step]), dtype=torch.int64)
            logit, node_embed, graph_embed = model(data.x, data.edge_index, data.batch)
            att_id, _, _, g_emb, g_emb_inv = explainModel(node_id, node_embed, batch, step)
            att_list.append(att_id)
            out1 = classifier(graph_embed)
            out2 = classifier(g_emb)
            out3 = classifier(g_emb_inv)
            if out1.argmax(dim=1)==out2.argmax(dim=1):
                if out1.argmax(dim=1)==data.y:
                    correct_list.append(step)
            c, num = accuracy(out2, out1)
            correct += c
            count += num
    print(f'The accuracy: {correct / count}.')
    open_file = open('ba2motif_motif_attention', 'wb')
    pickle.dump(att_list, open_file)
    open_file.close()
    return correct / count, correct_list

def train_node(data, model, criterion, optimizer, explainModel):
    collect = 0
    correct = 0
    l = 0.0
    attention = []
    attention_test = []
    min_loss = 10.0
    for i in range(len(data)):
        batch = torch.zeros(data[i].x.size()[0], dtype=torch.int64)
        logit, node_embed = model(data[i].x, data[i].edge_index, batch)
        pred = logit.argmax(dim=1)
        collect += 1
        explainModel.train()
        out, att = explainModel(node_embed)
        attention.append(att)
        result = out.argmax(dim=1)
        loss = criterion(out, pred)
        l += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if result == pred:
            correct += 1
    if (l / collect) < min_loss:
        min_loss = (l / collect)
        att_emb = explainModel.att_embedding.weight.data.cpu()
        open_file = open('node_attention_emb_mutag_mean', 'wb')
        pickle.dump(att_emb, open_file)
        open_file.close()
        open_file = open('node_attention_mutag_mean', 'wb')
        pickle.dump(attention, open_file)
        open_file.close()
    print(correct)
    print(f'The final loss: {l / collect}. The accuracy: {correct / collect}.')

def train_att_pool(data, model, criterion, optimizer, explainModel):
    collect = 0
    correct = 0
    correct_test = 0
    m_test = 0
    l = 0.0
    attention = []
    attention_test = []
    min_loss = 10.0
    explainModel.train()
    for i in range(3000):
        batch = torch.zeros(data[i].x.size()[0], dtype=torch.int64)
        logit, node_embed = model(data[i].x, data[i].edge_index, batch)
        pred = logit.argmax(dim=1)
        out, att = explainModel(node_embed)
        attention.append(att)
        result = out.argmax(dim=1)
        if result == data[i].y:
            collect += 1
        if pred == data[i].y:
            correct += 1
        loss = criterion(out, data[i].y)
        l += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    explainModel.eval()
    with torch.no_grad():
        for i in range(3000, len(data)):
            batch = torch.zeros(data[i].x.size()[0], dtype=torch.int64)
            logit, node_embed = model(data[i].x, data[i].edge_index, batch)
            pred = logit.argmax(dim=1)
            if pred == data[i].y:
                m_test += 1
            out, att = explainModel(node_embed)
            result = out.argmax(dim=1)
            if result == data[i].y:
                correct_test += 1
    print(f'Training accuracy: {correct / 3000}. Testing accuracy: {m_test / 1337}.')


model = GNN2(input_channels=10, hidden_channels=128, output_channels=2)

model2 = GNN2(input_channels=10, hidden_channels=128, output_channels=2)
classifier = nn.Linear(128, 2)
classifier.load_state_dict(torch.load('model/mlp_ba2motif'))
classifier.eval()
explainModel = AttExplainer(128)

for param_tensor in explainModel.state_dict():
    print(param_tensor, "\t", explainModel.state_dict()[param_tensor].size())
optimizer = torch.optim.Adam(explainModel.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
model.load_state_dict(torch.load('model/gcn_ba2motif'))
model2.load_state_dict(torch.load('model/gcn_ba2motif'))
model.eval()
model2.eval()

classifier.eval()
best_acc = 0.0
best_loss = 1000
min_loss = 100.0
for epoch in tqdm(range(500), desc='Epoch'):

    min_loss, acc = train(model, model2, optimizer, explainModel, classifier, criterion, min_loss)
    if min_loss <= best_loss:
        best_loss = min_loss

explainModelTest = AttExplainer(128)
explainModelTest.load_state_dict(torch.load('model/gcn_ba2motif_motif_explain'))
test_acc, correct_list = test(model, explainModelTest, classifier)
open_file = open('ba2motif_motif_correct_list', 'wb')
pickle.dump(correct_list, open_file)
open_file.close()
print(f'Best loss is {best_loss}, best acc is {test_acc}.')
