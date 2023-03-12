import os
from torch.utils.data import Dataset
from utils._pil import *
from pathlib import Path
import pandas as pd

# path = Path('./foodx')

class Foodx(Dataset):
    def __init__(self, path, transform):
      
        self.path = path
        
        data_train = pd.read_csv(path/"annot/train_info.txt", header=None, sep=" ")
        data_train.columns = ["name", "label"]
        data_train['name']='train/'+data_train['name']

        data_val = pd.read_csv(path/"annot/val_info.txt", header=None, sep=" ")
        data_val.columns = ["name", "label"]
        data_val['name']='val/'+data_val['name']

        data_test = pd.read_csv(path/"annot/test_info.txt", header=None, sep=" ")
        data_test.columns = ["name", "label"]
        data_test['name']='test/'+data_test['name']

        data = pd.concat([data_train, data_val, data_test])

        self.transform = transform
        self.images = data['name']
        self.labels = data['label']
       
        self.num_files = self.labels.shape[0]      
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index]
        
        file_name = self.images.iloc[index]
        path = self.path/self.phase/file_name
        x = read_image(path)
        try:
            x = self.transform(x)
        except:
            x = Image.open(path).convert('RGB')
            x = self.transform(x)
        # print(path)
        return x,y