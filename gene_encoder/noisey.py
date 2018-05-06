import torch
import torch.nn as nn


class Noisey(nn.Module):

    def __init__(self, features, alpha, beta):
        super(Noisey, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.reg = nn.Softmax(1)

        self.window3 = nn.Conv1d(features, 12, 3, padding=1)

        self.act1 = nn.Sigmoid()

        self.stride1 = nn.Conv1d(12, 12, 3, padding=1, stride=2)

        self.act2 = nn.Sigmoid()

        self.window5 = nn.Conv1d(12, 6, 5, padding=2)

        self.act3 = nn.Sigmoid()

        self.stride2 = nn.Conv1d(6, 6, 5, padding=2, stride=2)

        self.act4 = nn.Sigmoid()

        self.var = nn.Conv1d(6, 1, 1)

        self.mu = nn.Conv1d(6, 1, 1)

        self.unrandom = nn.ConvTranspose1d(1, 6, 1)

        self.randact = nn.Sigmoid()

        self.unstride1 = nn.ConvTranspose1d(6, 6, 5, padding=2, stride=2)

        self.act5 = nn.Sigmoid()

        self.dewindow5 = nn.ConvTranspose1d(6, 12, 5, padding=2)

        self.act6 = nn.Sigmoid()

        self.unstride2 = nn.ConvTranspose1d(12, 12, 3, padding=1, stride=2)

        self.act7 = nn.Sigmoid()

        self.dewindow3 = nn.ConvTranspose1d(12, features, 3, padding=1)

        self.softmax = nn.Softmax(1)

    def forward(self, genes, masks):
        # tmp = self.reg(genes + torch.normal(self.alpha * masks, self.beta * masks))
 
        tmp = genes
        tmp = self.act1(self.window3(tmp))
        tmp = self.act2(self.stride1(tmp))
        tmp = self.act3(self.window5(tmp))
        tmp = self.act4(self.stride2(tmp))
        var = self.var(tmp)
        mu = self.mu(tmp)
        tmp = torch.normal(mu, var)
        tmp = self.randact(self.unrandom(tmp))
        tmp = self.act5(self.unstride1(tmp))
        tmp = self.act6(self.dewindow5(tmp))
        tmp = self.act7(self.unstride2(tmp))
        tmp = self.softmax(self.dewindow3(tmp))

        return tmp
