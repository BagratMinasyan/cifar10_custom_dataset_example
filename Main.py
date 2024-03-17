import pandas as pd
import torch
from torch import nn
from Dataset import Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

path = "D:\\pytorch\\dataloader\\data\\"
file='annotations.csv'
data=Data(path, file, transforms.ToTensor())
data_generator=DataLoader(data, batch_size=2)

i=0
for x,y in data_generator:
    print(y)
    if i==1:
        break
    i+=1