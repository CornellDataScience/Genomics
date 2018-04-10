import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import autograd


basemap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


class MaizeGeneDataset(Dataset):

    def __init__(self, inputs_file, targets_file, group):
        """

        :param inputs_file:
        :param targets_file:
        """
        x = pd.read_csv(inputs_file)
        y = pd.read_csv(targets_file)
        self.inputs = x.columns.tolist()[2:]
        self.targets = y.columns.tolist()[2:]
        self.data = pd.merge(x, y, on=["gene_id", "group"])
        self.data = self.data.loc[self.data['group'] == group]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data.iloc[i].to_dict()

    def get_loader(self, batch_size=32, num_workers=8):
        return DataLoader(self, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)

    def pad_load(self, dataloader, genes, targets):
        ret = []
        for batch in dataloader:
            nw = {}
            for col in genes:
                nw[col] = self.pad_genes(batch[col])
            for col in targets:
                nw[col] = batch[col]
            ret.append(nw)
        return ret

    @staticmethod
    def pad_genes(genes):
        lens = torch.LongTensor([i for i in map(lambda s: len(s), genes)])
        maxlen = max(lens)
        padded = torch.zeros(len(genes), maxlen, 4)

        for i, g in enumerate(genes):
            for j, b in enumerate(g):
                if b in basemap:
                    padded[i, j, basemap[b]] = 1

        end_inds = lens - 1

        return padded
