from platform import node
from networkx.algorithms.shortest_paths.unweighted import predecessor
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import FloatTensor as FT
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GraphConv, BatchNorm
from torch_geometric.data import Data
import math
device = torch.device('cpu')
class GNN1(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GNN1, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(input_channels, hidden_channels)  # TODO
        self.conv2 = GraphConv(hidden_channels, hidden_channels)  # TODO
        self.conv3 = GraphConv(hidden_channels, hidden_channels)  # TODO
        self.lin = Linear(hidden_channels, output_channels)
        self.bn = BatchNorm(hidden_channels)
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        conv1_embed = x
        x = self.conv2(x, edge_index)
        x = x.relu()
        conv2_embed = x
        x = self.conv3(x, edge_index)
        x = x.relu()
        node_embed = x
        graph_embed = x
        x = self.drop_layer(x)
        x = self.lin(x)
        
        return x, node_embed

class GNN2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GNN2, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GCNConv(input_channels, hidden_channels)  # TODO
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # TODO
        self.conv3 = GCNConv(hidden_channels, hidden_channels)  # TODO
        self.lin = Linear(hidden_channels, output_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)
        conv1_embed = x
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.bn2(x)
        conv2_embed = x
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.bn3(x)
        node_embed = x
        x = global_mean_pool(x, batch.to(device))
        graph_embed = x
        x = self.drop_layer(x)
        x = self.lin(x)
        
        return x, node_embed, graph_embed

class AttLayer(torch.nn.Module):
    def __init__(self, emb_dimension):
        super(AttLayer, self).__init__()
        self.emb_size = 1
        self.emb_dimension = emb_dimension
        self.att_embedding = nn.Embedding(self.emb_size, self.emb_dimension)
        self.att_embedding.weight = nn.Parameter(FT(1, self.emb_dimension).uniform_(-0.5, 0.5))
        self.att_embedding.weight.requires_grad = True
        self.m = torch.nn.Softmax(dim=0)
    
    def forward(self, x, batch):
        indices_list = []
        indices_list.append(0)
        number_of_nodes = batch.tolist()[-1]
        count = 1
        for i in range(len(batch.tolist())):
            if batch.tolist()[i] == count:
                indices_list.append(i)
                count += 1
        indices_list.append(len(batch.tolist()))
        g_emb = []
        for i in range(len(indices_list)-1):
            n_emb = x[indices_list[i]:indices_list[i+1]]
            att_list = []
            input = torch.LongTensor([0])
            att_embedding = self.att_embedding(input)
            embedding_list = []
            for i in range(n_emb.size()[0]):
                embedding_list.append(n_emb[i])
            for i in embedding_list:
                result = torch.matmul(i, torch.t(att_embedding))
                att_list.append(result)
            att_list = self.m(torch.stack(att_list)).squeeze()

            mid_result = []
            for i in range(len(att_list)):
                mid_result.append(torch.mul(embedding_list[i], att_list[i]))
            graph_emb = torch.mean(torch.stack(mid_result), 0, keepdim=True)
            g_emb.append(graph_emb)
        g_emb = torch.stack(g_emb).squeeze()
        return g_emb, self.att_embedding.weight.data.cpu()

class AttGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(AttGNN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(input_channels, hidden_channels)  # TODO
        self.conv2 = GraphConv(hidden_channels, hidden_channels)  # TODO
        self.conv3 = GraphConv(hidden_channels, hidden_channels)  # TODO
        self.lin = Linear(hidden_channels, output_channels)
        self.attPool = AttLayer(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x, att = self.attPool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class FCL(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FCL, self).__init__()
        self.fcl = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.7)

    def forward(self, gcn_output):
        output = self.fcl(gcn_output)
        return output

class NodeAttExplainer(nn.Module):
    def __init__(self, emb_dimension):
        super(NodeAttExplainer, self).__init__()
        self.emb_dimension = emb_dimension
        self.att_embedding = nn.Parameter(torch.ones(self.emb_dimension))
        self.att_embedding.requires_grad = True
        self.m = torch.nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, n_emb, batch):
        a = torch.matmul(n_emb, self.att_embedding)
        a = self.m(a/3)
        b = torch.t(n_emb) * a
        b = torch.t(b)
        return a, global_add_pool(b, batch)

class AttExplainer(nn.Module):
    def __init__(self, emb_dimension):
        super(AttExplainer, self).__init__()
        self.emb_dimension = emb_dimension
        self.att_embedding = nn.Parameter(torch.randn(self.emb_dimension))
        self.trans = nn.Sequential(nn.Linear(self.emb_dimension, 256), nn.ReLU(), nn.Linear(256, self.emb_dimension))
        
        self.att_embedding.requires_grad = True
        self.m = torch.nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, node_id, n_emb, batch, step):
        all_embed_list = []

        all_embed = torch.zeros(128).unsqueeze(0)
        all_a = torch.zeros(1)
        all_c = torch.zeros(1)
        embed_list = []
        for i in node_id[step]:
            mask = torch.zeros(n_emb.size()[0], dtype=bool)

            for j in i:
                mask[j] = True
            embed_list.append(torch.mean(n_emb[mask, :], 0).unsqueeze(dim=0))
        embed = torch.cat(embed_list, dim=0)

        # attention = self.trans(self.att_embedding)
        attention = self.att_embedding
        # print(attention.size())
        # print(stop)
        a = torch.matmul(embed, attention)
        a = self.m(a/10)

        inv_a = 1 - a

        att_ori = torch.clone(a).tolist()
        att_ori.sort(reverse=True)
        cut_line = 0.1
        
        c = self.sig(1000*(a-cut_line))
        inv_c = 1 - c
        b = torch.t(embed) * a
        b = torch.t(b)

        
        inv_b = torch.t(embed) * inv_a
        inv_b = torch.t(inv_b)
        
        new_emb = torch.t(embed) * c
        new_emb = torch.t(new_emb)

        return a, embed, global_mean_pool(new_emb, batch), global_add_pool(b, batch), global_add_pool(inv_b, batch)
            
class NewAttExplainer(nn.Module):
    def __init__(self, emb_dimension):
        super(NewAttExplainer, self).__init__()
        self.emb_dimension = emb_dimension
        self.att_embedding = nn.Parameter(torch.randn(self.emb_dimension))
        self.att_embedding.requires_grad = True
        self.w = nn.Parameter(torch.randn(self.emb_dimension, self.emb_dimension))
        self.w.requires_grad = True
        self.m = torch.nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.norm = BatchNorm(self.emb_dimension)
    
    def forward(self, all_embed, ori_embed, all_final_batch, step):
        embed = all_embed[step].to(device)
        final_batch = all_final_batch[step].to(device)
        a = torch.matmul(embed.to(device), self.w.to(device))
        a = torch.matmul(a, torch.t(ori_embed))

        a = torch.t(self.m(a/0.05)).to(device)

        b = torch.t(embed) * a
        b = torch.t(b)
        cut_line = 0.11
        
        c = self.sig(-100000000*(a-0.0000001))
        new_emb = torch.t(embed) * c
        new_emb = torch.t(new_emb)

        return a, global_mean_pool(new_emb, final_batch), global_add_pool(b, final_batch)
            
class BashapeAttExplainer(nn.Module):
    def __init__(self, emb_dimension):
        super(BashapeAttExplainer, self).__init__()
        self.emb_dimension = emb_dimension
        self.att_embedding = nn.Parameter(torch.randn(self.emb_dimension))
        self.att_embedding.requires_grad = True
        self.w = nn.Parameter(torch.randn(self.emb_dimension, self.emb_dimension))
        self.w.requires_grad = True
        self.w2 = nn.Parameter(torch.randn(self.emb_dimension, self.emb_dimension))
        self.w2.requires_grad = True
        self.m = torch.nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.norm = BatchNorm(self.emb_dimension)
    
    def forward(self, embed, target_embed):
        a = torch.matmul(embed, self.att_embedding)
        a = torch.t(self.m(a/1))
        b = torch.t(embed) * a
        b = torch.t(b)
        
        batch = torch.zeros(b.size()[0], dtype=torch.int64)
        return a, global_add_pool(b, batch)
