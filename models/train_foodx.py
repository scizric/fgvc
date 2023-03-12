from datasets.foodx import Foodx
from models._transforms import give_transform
from pathlib import Path

path = Path('./datasets/data/Foodx-251')
transform=give_transform()
dataset = Foodx(path, transform)

from utils.configs import *
from models.model import *
from models.Kfold import run_Kfold

model_type=32
num_classes=251
model,model_name=give_model(model_type,num_classes)
run_Kfold(dataset, model, 'foodx_'+model_name, num_epochs)