from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io
import numpy as np
class Data(Dataset):
    def __init__(self, root_dir, csv, transform=None):
        super().__init__()
        self.annotations=pd.read_csv(csv)
        self.root_dir=root_dir
        self.transform =transform
    def __len__(self, ):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path=self.annotations.iloc[index,0]
        print(img_path)
        x=io.imread(img_path)
        y=torch.Tensor(int(self.annotations.iloc[index,2]))

        if self.transform:
            x=self.transform(x)
        x=torch.transpose(x,2,0)
        x=x.unsqueeze(0)
        print(x.shape)
        return x,y