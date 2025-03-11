import torch

from torch_geometric.utils.num_nodes import maybe_num_nodes

class ReadOut(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def divided_graph(x, batch_index):
        graph = []
        for i in range(batch_index[-1] + 1):
            graph.append(x[batch_index == i])

        return graph

    def forward(self, x: torch.tensor, batch_index) -> torch.tensor:
        graph = ReadOut.divided_graph(x, batch_index)

        for i in range(len(graph)):
            graph[i] = graph[i].mean(dim=0).unsqueeze(0)

        out_readoout = torch.cat(graph, dim=0)

        return out_readoout


def normalize(x: torch.Tensor):
    x -= x.min()
    if x.max() == 0:
        return torch.zeros(x.size(), device=x.device)
    x /= x.max()
    return x


def subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
    else:
        node_idx = node_idx.to(row.device)


    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break



    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask