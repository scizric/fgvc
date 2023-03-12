from datasets.cub import CUB_data, CUB
from models._transforms import give_transform
from pathlib import Path

path = Path('./datasets/data/CUB_200_2011')
data=CUB_data(path)
transform=give_transform()
dataset = CUB(path, data, transform)

from utils.configs import *
from models.model import *
from models.Kfold import run_Kfold

model_type=32
num_classes=200
model,model_name=give_model(model_type,num_classes)

run_Kfold(dataset, model, 'cub_'+model_name, num_epochs)