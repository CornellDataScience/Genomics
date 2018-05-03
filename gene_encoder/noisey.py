import torch
import torch.nn as nn


class Noisey(nn.Module):

    def __init__(self, features, wchs, enc_size, alpha, beta):
        super(Noisey, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.reg = nn.Softmax(1)

        self.window1 = nn.Conv1d(features, wchs[0], 1, padding=0)
        self.window3 = nn.Conv1d(features, wchs[1], 3, padding=1)
        self.window5 = nn.Conv1d(features, wchs[2], 5, padding=2)
        self.window7 = nn.Conv1d(features, wchs[3], 7, padding=3)
        self.window9 = nn.Conv1d(features, wchs[4], 9, padding=4)

        self.act1 = nn.Sigmoid()

        self.enc = nn.LSTM(
            input_size=sum(wchs),
            hidden_size=enc_size / 2,
            batch_first=True,
            bidirectional=True
        )

        self.act2 = nn.Sigmoid()

        self.dewindow1 = nn.Conv1d(enc_size, wchs[0], 1, padding=0)
        self.dewindow3 = nn.Conv1d(enc_size, wchs[1], 3, padding=1)
        self.dewindow5 = nn.Conv1d(enc_size, wchs[2], 5, padding=2)
        self.dewindow7 = nn.Conv1d(enc_size, wchs[3], 7, padding=3)
        self.dewindow9 = nn.Conv1d(enc_size, wchs[4], 9, padding=4)

        self.to_bases = nn.Conv1d(sum(wchs), 1, 1)

    def forward(self, genes, masks):
        tmp = self.reg(genes + torch.normal(self.alpha * masks, self.beta * masks))
        tmp = self.act1(torch.cat((
            self.window1(tmp),
            self.windows3(tmp),
            self.windows5(tmp),
            self.windows7(tmp),
            self.windows9(tmp)
        ), 1))
        tmp, _ = self.enc(tmp)
        tmp = self.act2(tmp)
        tmp = self.to_bases(torch.cat((
            self.dewindow1(tmp),
            self.dewindow3(tmp),
            self.dewindow5(tmp),
            self.dewindow7(tmp),
            self.dewindow9(tmp)
        )))

        return self.to_bases(tmp)

    def encode(self, genes):
        tmp = self.act1(torch.cat((
            self.window1(genes),
            self.windows3(genes),
            self.windows5(genes),
            self.windows7(genes),
            self.windows9(genes)
        ), 1))
        tmp, (_, c) = self.enc(tmp)
        return c
