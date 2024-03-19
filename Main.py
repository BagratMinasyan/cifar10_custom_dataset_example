import pandas as pd
import torch
from torch import nn
from Dataset import Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

path = "D:\\pytorch\\dataloader\\data\\"
file='annotations.csv'
data=Data(path, file)
batch_size=8
data_generator=DataLoader(data, batch_size=batch_size)

x,y = next(iter(data_generator))
for i in range(x.shape[0]):
    plt.subplot(2,4,i+1)
    plt.imshow(x[i].permute(1,2,0))
plt.show()

