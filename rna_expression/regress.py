import torch
import torch.nn as nn
from util.fire import Fire1d


class RNARegress(nn.Module):

    def __init__(self,
                 features=4,
                 enc_size=256,
                 layers=None,
                 lin_features=None,
                 out_size=1):
        super(RNARegress, self).__init__()

        if not layers:
            layers = []

        self.enc_gru = nn.GRU(features,
                              enc_size,
                              batch_first=True,
                              bidirectional=True)

        self.layers = []

        for name, ltype, params in layers:
            if ltype == "fire":
                self.add_fire(name, params)
            elif ltype == "pool":
                self.add_pool(name, params)
            elif ltype == "conv":
                self.add_conv(name, params)
            elif ltype == "norm":
                self.add_norm(name, params)
            elif ltype == "relu":
                self.add_relu(name, params)

        self.out = nn.Linear(lin_features, out_size)

    def forward(self, genes):
        x = []
        for g in genes:
            tmp, _ = self.enc_gru(g)
            tmp = tmp[:, -1, :]
            x.append(tmp)
        x = torch.stack(x, 1)
        for name in self.layers:
            x = getattr(self, name)(x)
        x = self.out(x)
        return x

    def add_fire(self, name, params):
        in_channel, s, e1, e3 = params
        setattr(self, name, Fire1d(in_channel, s, e1, e3))
        self.layers.append(name)

    def add_pool(self, name, params):
        kern, stride = params
        setattr(self, name, nn.MaxPool1d(kern, stride))
        self.layers.append(name)

    def add_conv(self, name, params):
        in_channel, c, kern, stride, pad = params
        setattr(self, name, nn.Conv1d(in_channel, c, kern, stride, pad))
        self.layers.append(name)

    def add_norm(self, name, params):
        num_features, = params
        setattr(self, name, nn.BatchNorm1d(num_features))
        self.layers.append(name)

    def add_relu(self, name, _):
        setattr(self, name, nn.ReLU())
        self.layers.append(name)
