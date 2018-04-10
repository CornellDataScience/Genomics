import torch
import torch.nn as nn


class Fire1d(nn.Module):

    def __init__(self, in_channel, s, e1, e3):
        super(Fire1d, self).__init__()

        self.squeeze = nn.Conv1d(in_channel, s, 1)
        self.squeeze_act = nn.ReLU()
        self.expand1 = nn.Conv1d(s, e1, 1)
        self.expand3 = nn.Conv1d(s, e3, 3, padding=1)
        self.expand_act = nn.ReLU()

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_act(x)
        x = torch.cat((self.expand1(x), self.expand3(x)), 1)
        x = self.expand_act(x)
        return x

