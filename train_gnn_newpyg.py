import torch
import torch.nn as nn
from utils.model import GNN2


import torch
from torch_geometric.datasets import TUDataset
device = torch.device('cpu')
dataset = TUDataset(root='TUDataset', name='NCI1')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]

print()
print(data)
print('=============================================================')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.contains_isolated_nodes()}')
print(f'Has self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(1)
dataset = dataset.shuffle()

train_dataset = dataset[:3500]
test_dataset = dataset[3500:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = GNN2(input_channels=dataset.num_features, hidden_channels=64, output_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader: 
         out, _, _ = model(data.x, data.edge_index, data.batch) 
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:
         out, _, _ = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

max_acc = 0.0
for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if train_acc >= max_acc:
        max_acc = train_acc
        torch.save(model.state_dict(), 'model/gcn_nci1')
        torch.save(model.lin.state_dict(), 'model/mlp_nci1')
        torch.save(model.conv2.state_dict(), 'model/conv2_nci1')
        torch.save(model.conv3.state_dict(), 'model/conv3_nci1')
        Final_train_acc = train_acc
        print(f'Best train acc is {train_acc}, the best test acc is {test_acc}.')
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')