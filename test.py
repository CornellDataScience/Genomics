from util import maize_gene_dataset
from rna_expression.regress import RNARegress
import torch
from torch import autograd, nn, optim

root = "./assets/"

inputs_file = "X.csv"
outputs_file = "y.csv"

train = maize_gene_dataset.MaizeGeneDataset(root + inputs_file,
                                            root + outputs_file,
                                            "train")

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

pad = train.pad_load(trainloader, ["TSS_promoter"], ["RNA_Leaf_Zone_3_Growth"])

net = RNARegress(features=4,
                 enc_size=128,
                 layers=[
                     ("conv1", "conv", (1, 32, 7, 1, 3)),
                     ("relu1", "relu", ()),
                     ("fire1", "fire", (32, 16, 64, 64)),
                     ("fire2", "fire", (128, 16, 64, 64)),
                     ("norm1", "norm", (128,)),
                     ("fire3", "fire", (128, 32, 128, 128)),
                     ("fire4", "fire", (256, 32, 128, 128)),
                     ("pool1", "pool", (7, 3)),
                     ("fire5", "fire", (256, 48, 192, 192)),
                     ("fire6", "fire", (384, 48, 192, 192)),
                     ("norm2", "norm", (384,)),
                     ("fire7", "fire", (384, 64, 256, 256)),
                     ("fire8", "fire", (512, 64, 256, 256)),
                     ("pool2", "pool", (4, 2)),
                     ("conv2", "conv", (512, 1, 11, 1, 5)),
                     ("relu2", "relu", ())
                 ],
                 lin_features=41,
                 out_size=1)

# print(net)

loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

epochs = 1

net.cuda()
# batches = [{"TSS_promoter": (torch.ones(2, 20, 4), torch.ones(2, out=torch.LongTensor(2))), "RNA_Leaf_Zone_3_Growth": torch.ones(2, 1)}]

for epoch in range(epochs):
    for i, batch in enumerate(pad):
        genes = batch["TSS_promoter"]
        targets = autograd.Variable(batch["RNA_Leaf_Zone_3_Growth"].cuda())
        outputs = net([autograd.Variable(genes.cuda())])

        optimizer.zero_grad()
        loss = loss_func(outputs, targets.float())
        loss.backward()
        optimizer.step()

        print("Loss: {}\n".format(loss.data[0]))
