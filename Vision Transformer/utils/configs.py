import torch
import torch.nn as nn
from sklearn.model_selection import KFold

device_gpu=torch.device('cuda:0')
device_cpu=torch.device('cpu')

#Training Parameters
batch_size=16
num_epochs=50
loss_func=nn.CrossEntropyLoss()

#5 fold cross validation
k=5
torch.manual_seed(12)
splits=KFold(n_splits=k,shuffle=True,random_state=12)

#Metrics store variables
fold_performance={}
fold_record = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}