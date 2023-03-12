from datasets.dogs import Dogs_data, Dogs
from models._transforms import give_transform
from pathlib import Path

path = Path('./datasets/data/Stanford_Dogs')
data=Dogs_data(path)
transform=give_transform()
dataset = Dogs(path, data, transform)

from utils.configs import *
from models.model import *
from models.Kfold import run_Kfold

model_type=32
num_classes=120
model,model_name=give_model(model_type,num_classes)
run_Kfold(dataset, model, 'dogs_'+model_name, num_epochs)