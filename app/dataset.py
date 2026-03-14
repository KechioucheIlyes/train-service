import numpy as np
import torch
from torch.utils.data import Dataset


class CovidTorchDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.df.iloc[idx]["image_array"]
        label = int(self.df.iloc[idx]["label_encoded"])

        img = np.array(img, dtype=np.float32)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if img.max() > 1.0:
            img = img / 255.0

        img = np.transpose(img, (2, 0, 1))

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )