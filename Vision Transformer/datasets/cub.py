from torch.utils.data import Dataset
from utils._pil import *
from pathlib import Path
import pandas as pd

# path = Path('./CUB_200_2011')

def CUB_data(path):
    labels = pd.read_csv(path/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]

    # train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
    # train_test.columns = ["id", "is_train"]

    images = pd.read_csv(path/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    classes = pd.read_csv(path/"classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]
    # categories = [x for x in classes["class"]]

    data=images.merge(labels, on='id')
    return data

class CUB(Dataset):
    def __init__(self, files_path, data, transform):
      
        self.files_path = files_path/'images'
        self.images = data[['id','name']]
        self.labels = data[['id','label']]

        self.transform = transform
        

        self.labels = self.labels
       
        self.num_files = self.labels.shape[0]      
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index,1] - 1
        
        file_name = self.images.iloc[index, 1]
        path = self.files_path/file_name
        x = read_image(path)
        try:
            x = self.transform(x)
        except:
            x = Image.open(path).convert('RGB')
            x = self.transform(x)
        # print(path)
        return x,y