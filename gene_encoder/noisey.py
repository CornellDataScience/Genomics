import torch
import torch.nn as nn


class Noisey(nn.Module):

    def __init__(self, features, wchs, enc_size, alpha, beta):
        super(Noisey, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.reg = nn.Softmax(1)

        self.window3 = nn.Conv1d(features, wchs, 4, padding=1)
        self.pooling = nn.MaxPool1d(4)
     
        self.act1 = nn.Sigmoid()

        self.enc = nn.LSTM(
            input_size=4,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.act2 = nn.Sigmoid()

        self.depooling = nn.MaxUnpool1d(3)

        self.dewindow3 = nn.Conv1d(enc_size, wchs, 4)
        self.to_bases = nn.Conv1d(wchs, 4, 1, )

        self.softmax = nn.Softmax()

    def forward(self, genes, masks):
        #tmp = self.reg(genes + torch.normal(self.alpha * masks, self.beta * masks))
 
        tmp = genes
        # print(tmp.shape)
        # tmp = self.window3(tmp)
        # tmp = self.pooling(tmp)
        
        tmp = tmp.transpose(1,2)
        
        print(tmp.shape)
        tmp, _ = self.enc(tmp)
       
        # tmp = self.act2(tmp.transpose(1,2))
        # tmp = self.dewindow3(tmp)
        print(tmp.shape)
        #tmp = self.to_bases(self.dewindow3(tmp))
     
      
      
        return tmp

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
