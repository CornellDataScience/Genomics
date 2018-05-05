from util import maize_gene_dataset as mgd
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pack_sequence
from torch import autograd, nn, optim
from gene_encoder.noisey import Noisey
from random import shuffle

root = "./assets/"

inputs_file = "X.csv"
outputs_file = "y.csv"

train = mgd.MaizeGeneDataset(root + inputs_file,
                                            root + outputs_file,
                                            "train", 'ATG_promoter', 'RNA_Leaf_Zone_3_Growth')
# def __init__(self, inputs_file, targets_file, group, inputs, targets, mode="match"):

# columns: ['gene_id', 'group',
#           'TSS_promoter', 'ATG_promoter',
#           'transcript_sequence', 'protein_sequence',
#           'RNA_Leaf_Zone_3_Growth',
#           'RNA_Root_Meristem_Zone_5_Days',
#           'RNA_X6_7_internode',
#           'RNA_X7_8_internode',
#           'Protein_Leaf_Zone_3_Growth',
#           'Protein_Root_Meristem_Zone_5_Days',
#           'Protein_X6_7_internode',
#           'Protein_X7_8_internode']

trainloader = train.get_loader(16, 8)

pad = train.pad_load(trainloader, ["TSS_promoter", "ATG_promoter"], ["RNA_Leaf_Zone_3_Growth"])

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
    for i, batch in enumerate(pad):
        genes = batch["TSS_promoter"]
        noise = torch.zeros(len(batch), 4, len(batch[0]), dtype=torch.float)
        genes = genes.cuda()
        noise = noise.cuda()
        outputs = net(genes, noise)

        optimizer.zero_grad()
        loss = loss_func(genes, outputs.data)
        train_error.append(loss)
        loss.backward()
        optimizer.step()
        logfile.write("Epoch {} Batch {} Train Loss: {}\n".format(epoch, i, loss.data))
        break
    break

    print(epoch)
    print("Train Loss: {}\n".format(loss.data))
    # if epoch > 0:
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


