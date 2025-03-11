import torch
from torch._C import GraphExecutorState
import torch.nn as nn
from torch import FloatTensor as FT
import pickle

from torch.nn.modules.loss import CrossEntropyLoss
from utils.model import GNN1, AttExplainer, NodeAttExplainer, BashapeAttExplainer
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


open_file = open('bashape', 'rb')
dataset = pickle.load(open_file)
open_file.close()
data = dataset[0]
batch = torch.zeros(data.num_nodes, dtype=torch.int64)

def accuracy(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    _, targets = torch.max(y, dim=1)
    return torch.sum(preds == targets).item(), len(preds)

def train(model, optimizer, explainModel, classifier, loss_function):
    open_file = open('node_id_bashape', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    open_file = open('new_node_id_bashapes', 'rb')
    new_node_id = pickle.load(open_file)
    open_file.close()
    open_file = open('bashape_embed', 'rb')
    motif_embed = pickle.load(open_file)
    open_file.close()
    l = 0.0
    ori_p = 0.0
    aft_p = 0.0
    count = 0.0
    min_loss = 10.0
    for i in range(80):
        att, n_embed = explainModel(motif_embed[i], node_embed[i*5+300, :])
        result = logit[i*5+300, :].argmax(dim=-1).unsqueeze(-1)
        logit1 = classifier(n_embed)
        ori_p += logit[i*5+300, result]
        aft_p += logit1[0, result]
        if result == logit1[0].argmax(dim=-1):
            count += 1
        loss = loss_function(logit1, result)
        l += loss
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    if l/80 < min_loss:
        min_loss = l/80
        torch.save(explainModel.state_dict(), 'model/gcn_bashape_explainer')
    print(f'Training acc is {count/80}. The loss is {l/80}.')
    return l/80, count/80

def test(model, explainModel, classifier):
    open_file = open('bashape_embed', 'rb')
    motif_embed = pickle.load(open_file)
    open_file.close()
    open_file = open('new_node_id_bashapes', 'rb')
    node_id = pickle.load(open_file)
    open_file.close()
    count = 0
    correct = 0
    att_list = []
    explainModel.eval()
    correct_list = []
    test_count = 0
    with torch.no_grad():
        for i in range(80):
            att, n_embed = explainModel(motif_embed[i], node_embed[i*5+300, :])
            if att.argmax(dim=-1) == torch.t(att).size()[0]-1:
                test_count += 1
    open_file = open('ba2motif_motif_attention', 'wb')
    pickle.dump(att_list, open_file)
    open_file.close()
    print(f'Test Accuracy is:{test_count/80}')
    return correct, correct_list

model = GNN1(input_channels=10, hidden_channels=64, output_channels=4)

classifier = nn.Linear(64, 4)
classifier.load_state_dict(torch.load('model/mlp_bashape_new'))
classifier.eval()
explainModel = BashapeAttExplainer(64)

optimizer = torch.optim.Adam(explainModel.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
model.load_state_dict(torch.load('model/gcn_bashape_new'))
model.eval()
logit, node_embed = model(data.x, data.edge_index)

classifier.eval()
best_acc = 0.0
best_loss = 1000
for epoch in tqdm(range(10), desc='Epoch'):

    loss, acc = train(model, optimizer, explainModel, classifier, criterion)
    if loss <= best_loss:
        best_loss = loss

explainModelTest = BashapeAttExplainer(64)
explainModelTest.load_state_dict(torch.load('model/gcn_bashape_explainer'))
explainModelTest.eval()
test_acc, correct_list = test(model, explainModelTest, classifier)