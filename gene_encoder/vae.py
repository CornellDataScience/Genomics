import torch
import torch.nn as nn


class Vae(nn.Module):

    def __init__(self, input_size, hidden_size, enc_size):
        super(Vae, self).__init__()

   
        self.enc_size = enc_size
        self.hidden = hidden_size

       
        
        self.c1 = nn.Conv1d(input_size, self.hidden, 2)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(self.hidden, self.hidden, 1)
        self.p2 = nn.AvgPool1d(2)
        self.mu = nn.Linear(5988, self.enc_size)
        self.var = nn.Linear(5988, self.enc_size)

        self.dc1 = nn.ConvTranspose1d(self.enc_size, self.enc_size, 1)
    

        self.act2 = nn.Sigmoid()




    def encode(self, genes):
        print(genes.shape)
        tmp = self.e1(genes)
        print(tmp.shape)
        tmp = self.bn1(tmp)
        return tmp


    def decode(self, z):
        pass

    def forward(self, genes, masks):
        print(genes.shape)
        #tmp = genes.transpose(0, 1).transpose(1, 2)
        tmp = self.c1(genes)
        print(tmp.shape)
        tmp = self.p1(tmp)
        tmp = self.c2(tmp)
        tmp = self.p2(tmp)
        print(tmp.shape)
        tmp = tmp.transpose(0,1)
        mu = self.mu(tmp)
        print(mu.shape)
        return tmp
