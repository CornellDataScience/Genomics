import pandas as pd
from torch.utils.data import Dataset


class MaizeGeneDataset(Dataset):

    def __init__(self, inputs_file, targets_file, group, inputs, targets, mode="match"):
        """

        :param inputs_file:
        :param targets_file:
        """
        x = pd.read_csv(inputs_file, usecols=['gene_id', 'group', *inputs])
        y = pd.read_csv(targets_file, usecols=['gene_id', 'group', *targets])
        self.targets = targets
        self.data = pd.merge(x, y, on=["gene_id", "group"])
        self.data = self.data.loc[self.data['group'] == group]
        if mode == "match":
            self.data = self.data
            self.inputs = inputs
        elif mode == "melt":
            self.data = pd.melt(self.data, value_vars=inputs, value_name="sequence", var_name="type")
            self.inputs = ["sequence"]
        elif mode == "cat":
            self.data["sequence"] = self.data[inputs].apply(lambda row: ''.join(row), axis=1)
            self.data.drop(columns=inputs)
            self.inputs = ["sequence"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data.iloc[i].to_dict()
