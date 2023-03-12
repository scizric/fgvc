import datetime as dt
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.optim as optim
from utils.configs import *
from models.train import *

def run_Kfold(dataset, model, model_name, num_epochs):
    with open(model_name,'w') as f:
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

            f.write('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            
            model.to(device_gpu)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            test_loss, test_correct=valid_epoch(model,device_gpu,test_loader,loss_func)
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            for epoch in range(num_epochs):
                train_loss, train_correct=train_epoch(model,device_gpu,train_loader,loss_func,optimizer)
                test_loss, test_correct=valid_epoch(model,device_gpu,test_loader,loss_func)

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler) * 100
                f.write("{} - Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(dt.datetime.now(),epoch + 1, num_epochs, train_loss, test_loss, train_acc, test_acc))
                
            fold_record['train_loss'].append(train_loss)
            fold_record['test_loss'].append(test_loss)
            fold_record['train_acc'].append(train_acc)
            fold_record['test_acc'].append(test_acc)
                

        avg_train_loss = np.mean(fold_record['train_loss'])
        avg_test_loss = np.mean(fold_record['test_loss'])
        avg_train_acc = np.mean(fold_record['train_acc'])
        avg_test_acc = np.mean(fold_record['test_acc'])

        f.write('Performance of {} fold cross validation'.format(k))
        f.write("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc))  