import pickle
import torch
import os.path as osp
from torch_geometric.data import Dataset, Data

class BA2Motif(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BA2Motif, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['BA-2motif.pkl']

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(idx) for idx in range(1000)]

    def download(self):
        pass

    def process(self):
        idx = 0
        with open(self.raw_paths[0], 'rb') as f:
            adjs, features, labels = pickle.load(f)
        adjs = torch.from_numpy(adjs)
        features = torch.from_numpy(features)
        labels = torch.argmax(torch.from_numpy(labels), dim=1)
        for i in range(1000):
            x = features[i]
            edge_index = (adjs[i]>0).nonzero().t()
            y = labels[i]
            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data