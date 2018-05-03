import torch.nn as nn


class AutoGene(nn.Module):

    def __init__(self,
                 features=4,
                 enc_size=256):
        super(AutoGene, self).__init__()

        self.enc = nn.LSTM(
            input_size=features,
            hidden_size=enc_size,
            batch_first=True,
            num_layers=2,
            bidirectional=True
        )

        self.dec = nn.LSTM(
            input_size=enc_size * 2,
            hidden_size=features,
            batch_first=True,
            num_layers=1,
            bidirectional=False
        )

    def forward(self, genes):
        output, _ = self.enc(genes)
        return self.dec(output)[0]
