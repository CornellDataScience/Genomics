import torch
import torch.nn as nn


class Noisey(nn.Module):

    def __init__(self, features, alpha, beta):
        super(Noisey, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.reg = nn.Softmax(1)

        self.window3 = nn.Conv1d(features, 8, 3, padding=1, stride=2)

        self.act1 = nn.ReLU()

        self.window5 = nn.Conv1d(8, 4, 5, padding=2, stride=2)

        self.act2 = nn.ReLU()

        self.stride1 = nn.Conv1d(4, 2, 5, padding=2, stride=2)

        self.act3 = nn.ReLU()

        self.stride2 = nn.Conv1d(2, 1, 7, padding=3, stride=2)

        self.act4 = nn.ReLU()

        self.unstride1 = nn.ConvTranspose1d(1, 2, 7, padding=3, stride=2)

        self.act5 = nn.ReLU()

        self.unstride2 = nn.ConvTranspose1d(2, 4, 5, padding=2, stride=2)

        self.act6 = nn.ReLU()

        self.dewindow5 = nn.ConvTranspose1d(4, 8, 5, padding=2, stride=2)

        self.act7 = nn.ReLU()

        self.dewindow3 = nn.ConvTranspose1d(8, features, 3, padding=1, stride=2)

        self.softmax = nn.Softmax(1)

    def forward(self, genes, masks):
        # tmp = self.reg(genes + torch.normal(self.alpha * masks, self.beta * masks))
 
        tmp = genes
        tmp = self.act1(self.window3(tmp))
        tmp = self.act2(self.window5(tmp))
        tmp = self.act3(self.stride1(tmp))
        tmp = self.act4(self.stride2(tmp))
        tmp = self.act5(self.unstride1(tmp))
        tmp = self.act6(self.unstride2(tmp))
        tmp = self.act7(self.dewindow5(tmp))
        tmp = self.softmax(self.dewindow3(tmp))

        return tmp
