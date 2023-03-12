import torch
import torch.nn as nn
from torchsummary import summary
from utils.configs import *

def give_model(model_type,num_classes):
    # 1 for convnext; 2 for vit and 3 for swin
    if model_type==1:
        model_name='convnext_tiny'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'convnext_tiny', pretrained=True)
        model.classifier[2]=nn.Linear(768,num_classes)
    elif model_type==12:
        model_name='convnext_base'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'convnext_base', pretrained=True)
        model.classifier[2]=nn.Linear(1024,num_classes)
    elif model_type==13:
        model_name='convnext_large'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'convnext_large', pretrained=True)
        model.classifier[2]=nn.Linear(1536,num_classes)

    elif model_type==2:
        model_name='vit_base16'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'vit_b_16', pretrained=True)
        model.heads.head=nn.Linear(768,num_classes)
    elif model_type==22:
        model_name='vit_large16'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'vit_l_16', pretrained=True)
        model.heads.head=nn.Linear(1024,num_classes)

    elif model_type==3:
        model_name='swin_t_tiny'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'swin_t', pretrained=True)
        model.head=nn.Linear(768,num_classes)
    elif model_type==32:
        model_name='swin_t_base'
        model = torch.hub.load('pytorch/vision:v0.14.0', 'swin_t', pretrained=True)
        model.head=nn.Linear(1024,num_classes)
    return model,model_name


# summary(model,(3,224,224))