from datasets.aircraft import Aircraft
from models._transforms import give_transform
from pathlib import Path

path = Path('./datasets/data/FGVC-Aircraft')
transform=give_transform()
dataset = Aircraft(path, transform)

from utils.configs import *
from models.model import *
from models.Kfold import run_Kfold

model_type=32
num_classes=100
model,model_name=give_model(model_type,num_classes)
run_Kfold(dataset, model, 'aircraft_'+model_name, num_epochs)