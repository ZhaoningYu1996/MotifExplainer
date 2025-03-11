import torch
import torch.nn as nn
from utils.model import GNN2
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pickle

import torch
# from test import BA2Motif
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree, dense_to_sparse

torch.manual_seed(12345)

class BA_Motifs(Dataset):
    def __init__(self, path='./data/BA2_Motifs/BA-2motif.pkl'):
        with open(path,'rb') as fin:
            adjs,features,labels = pickle.load(fin)
        self.adjs = adjs
        self.features = features
        self.labels = labels
        self.num_classes = 2

    def __getitem__(self, index):
        adj = self.adjs[index]
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        x = torch.from_numpy(self.features[index])
        y = torch.argmax(torch.from_numpy(self.labels[index]))
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def __len__(self):
        return len(self.labels)

dataset = BA_Motifs()
data = dataset[0]

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')

for i in range(10):
    print(dataset[i])
    print(dataset[i].x)

print()
print(data)
print('=============================================================')

# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.contains_isolated_nodes()}')
# print(f'Has self-loops: {data.contains_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(1)
dataset = dataset.shuffle()

print(f"Number of data: {len(dataset)}")
# print(stop)

train_dataset = dataset[:900]
test_dataset = dataset[900:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cpu')

model = GNN2(input_channels=dataset.num_features, hidden_channels=128, output_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    if test_acc >= max_acc:
        max_acc = test_acc
        torch.save(model.state_dict(), 'model/gcn_ba2motif')
        # torch.save(model.lin.state_dict(), 'model/mlp_mutagenicity_new')
        # torch.save(model.conv2.state_dict(), 'model/conv2_mutagenicity_new')
        # torch.save(model.conv3.state_dict(), 'model/conv3_mutagenicity_new')
        Final_train_acc = train_acc
        print(f'Best train acc is {train_acc}, the best test acc is {test_acc}.')
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')