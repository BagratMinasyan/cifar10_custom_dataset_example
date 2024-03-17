import os
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = "D:\\pytorch\\dataloader\\data\\"
dir_list = x = [os.path.abspath(os.path.join(path, p)) for p in os.listdir(path)]

with open('annotations.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['path','name'])
    for i in dir_list:
        string=i.split('_')[1]
        string=string.split('.')[0]
        writer = csv.writer(file)
        writer.writerow([i,string])

df=pd.read_csv('annotations.csv')
df['label'] = LabelEncoder().fit_transform(df.name.values)
df.to_csv('annotations.csv',index=None)