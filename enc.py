import util.maize_gene_dataset as mgd
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pack_sequence
from torch import autograd, nn, optim
from gene_encoder.autoencoder import AutoGene

inputs = ['ATG_promoter']
targets = ['RNA_Leaf_Zone_3_Growth']

dataset = mgd.MaizeGeneDataset('assets/X.csv',
                               'assets/y.csv',
                               'train',
                               inputs,
                               targets,
                               mode="melt")

valset = mgd.MaizeGeneDataset('assets/X.csv',
                              'assets/y.csv',
                              'val',
                              inputs,
                              targets,
                              mode="melt")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=True)


basemap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


def pack_batches(loader, set):
    adjusted = []
    for batch in loader:
        batch[set.inputs[0]] = sorted(batch[set.inputs[0]], key=lambda x: len(x), reverse=True)
        tmpbatch = []
        for gene in batch[set.inputs[0]]:
            tmpgene = []
            for b in gene:
                tmp = [0, 0, 0, 0]
                if b in basemap:
                    tmp[basemap[b]] = 1
                tmpgene.append(tmp)
            tmpbatch.append(tmpgene)
        var = torch.FloatTensor(tmpbatch)
        var = var.cuda()
        batch[set.inputs[0]] = pack_sequence(autograd.Variable(var))
        adjusted.append(batch[set.inputs[0]])
    return adjusted


data = pack_batches(dataloader, dataset)
val = pack_batches(valloader, valset)
del dataset
del dataloader
del valset
del valloader

net = AutoGene()
net = net.cuda()

epochs = 50
loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

train_error = []
val_error = []

logfile = open('log.txt', 'w+')

for epoch in range(epochs):
    for batch in data:
        outputs = net(batch)

        optimizer.zero_grad()
        loss = loss_func(outputs.data, batch.data)
        train_error.append(loss)
        loss.backward()
        optimizer.step()

        print("Train Loss: {}\n".format(loss.data))
        logfile.write("Epoch {} Train Loss: {}\n".format(epoch, loss.data))
    if epoch > 0 and epoch % 5 == 0:
        tmperror = 0
        numits = 0
        for batch in val:
            outputs = net(batch)
            loss = loss_func(outputs.data, batch.data)
            tmperror += loss
            numits += 1
        val_error.append(tmperror / numits)
        if len(val_error) > 0 and val_error[-1] > val_error[-2]:
            break
        logfile.write("\nValidation Loss: {}\n\n".format(loss.data))
    torch.save(net, "checkpoint.pt")


