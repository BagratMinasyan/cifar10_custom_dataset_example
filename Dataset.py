from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io
from skimage.transform import resize
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
        x=io.imread(img_path)
        x=resize(x,(32,32))
        y=torch.tensor(self.annotations.iloc[index,2])
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)
        if self.transform:
            x=self.transform(x)
        return x,y