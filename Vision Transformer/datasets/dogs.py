from torch.utils.data import Dataset
from utils._pil import *
from pathlib import Path
import pandas as pd

# path = Path('./Images')

def Dogs_data(path):
    # create a mapping from directory name to label index
    label2index = {n.name: i for i, n in enumerate(path.iterdir())}
    ids=[]
    fpaths = []
    labels = []
    id=0
    for dir in path.iterdir():
        for file in dir.iterdir():
            ids.append(id)
            labels.append(label2index[dir.name])
            fpaths.append(str(file))
            id=id+1

    data = pd.DataFrame(data={"id": ids, "name": fpaths, "label": labels})
    # data.head(2)
    return data


class Dogs(Dataset):
    def __init__(self, path, data, transform):
      
        self.path = path
        self.images = data[['id','name']]
        self.labels = data[['id','label']]

        self.transform = transform
        self.labels = self.labels
       
        self.num_files = self.labels.shape[0]      
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index,1]
        
        file_name = self.images.iloc[index, 1]
        path = self.path/file_name
        x = read_image(path)
        try:
            x = self.transform(x)
        except:
            x = Image.open(path).convert('RGB')
            x = self.transform(x)
        # print(path)
        return x,y