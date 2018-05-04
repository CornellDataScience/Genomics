import util.maize_gene_dataset as mgd
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pack_sequence
from torch import autograd, nn, optim
from gene_encoder.noisey import Noisey
from random import shuffle

inputs = ['ATG_promoter', 'transcript_sequence']
targets = []

dataset = mgd.MaizeGeneDataset('assets/X.csv',
                               'assets/y.csv',
                               'train',
                               inputs,
                               targets,
                               mode="melt")

# valset = mgd.MaizeGeneDataset('assets/X.csv',
#                               'assets/y.csv',
#                               'val',
#                               inputs,
#                               targets,
#                               mode="melt")


basemap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


def mycollate(batch):
    b = torch.zeros(len(batch), 4, len(batch[0]), requires_grad=True)
    m1 = torch.zeros(len(batch), 4, len(batch[0]), dtype=torch.float)
    m2 = torch.zeros(len(batch), 4, len(batch[0]), dtype=torch.uint8)
    for i, gene in enumerate(batch):
        for j, c in enumerate(gene):
            if c in basemap:
                b[i, basemap[c], j] = 1
                m1[i, :, j] = 1
            m2[i, :, j] = 1
    return b, m1, m2


print("start load")
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, collate_fn=mycollate)
# valloader = DataLoader(valset, batch_size=32, num_workers=8)
del dataset
print("end load")

# val = []
# for batch in valloader:
#     batch = batch['sequence']
#     b = torch.zeros(len(batch), 4, len(batch[0]), requires_grad=True)
#     m = torch.zeros(len(batch), 1, len(batch[0]), requires_grad=True)
#     for i, gene in enumerate(batch):
#         for j, c in enumerate(gene):
#             if c in basemap:
#                 b[i, basemap[c], j] += 1
#                 b[i, 1,  j] += 1
#     val.append((b, m))
#
# del valset
# del valloader

net = Noisey(4, [1, 12, 9, 6, 3], 64, 0.2, 0.1)
net = net.cuda()

epochs = 50
loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_error = []
val_error = []

logfile = open('log.txt', 'w+')
print("Starting\n")
for epoch in range(epochs):
    for i, (genes, m1, m2) in enumerate(dataloader):
        genes = genes.cuda()
        m1 = m1.cuda()
        outputs = net(genes, m1)

        optimizer.zero_grad()
        loss = loss_func(genes[m2], outputs.data[m2])
        train_error.append(loss)
        loss.backward()
        optimizer.step()

        print("Train Loss: {}\n".format(loss.data))
        logfile.write("Epoch {} Batch {} Train Loss: {}\n".format(epoch, i, loss.data))
    # if epoch > 0 and epoch % 5 == 0:
    #     tmperror = 0
    #     numits = 0
    #     for batch in val:
    #         batch = batch.cuda()
    #         outputs = net(batch)
    #         loss = loss_func(outputs.data, batch.data)
    #         tmperror += loss
    #         numits += 1
    #     val_error.append(tmperror / numits)
    #     if len(val_error) > 0 and val_error[-1] > val_error[-2]:
    #         break
    #     logfile.write("\nValidation Loss: {}\n\n".format(loss.data))
    torch.save(net, "checkpoint.pt")


