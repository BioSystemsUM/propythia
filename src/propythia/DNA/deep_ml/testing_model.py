import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils
import torch
import os
from torch.optim import Adam
from src.train import traindata
from src.test import test
from src.models import MLP
from torch import nn
from os.path import exists

def split_standardizer_dataloader(fps_x,fps_y,):
    print('Splitting datasets...')
    x, x_test, y, y_test = train_test_split(
        fps_x, fps_y,
        test_size=0.2,
        train_size=0.8,
        stratify=fps_y
    )
    x_train, x_cv, y_train, y_cv = train_test_split(
        x, y,
        test_size=0.25,
        train_size=0.75,
        stratify=y
    )

    print('Standardizing datasets...')
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_cv = scaler.transform(x_cv)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_cv.shape)
    print(y_cv.shape)
        
    train_data = data_utils.TensorDataset(
        torch.tensor(x_train, dtype=torch.float),
        torch.tensor(y_train.to_numpy(), dtype=torch.long)
    )
    test_data = data_utils.TensorDataset(
        torch.tensor(x_test, dtype=torch.float),
        torch.tensor(y_test.to_numpy(), dtype=torch.long)
    )
    valid_data = data_utils.TensorDataset(
        torch.tensor(x_cv, dtype=torch.float),
        torch.tensor(y_cv.to_numpy(), dtype=torch.long)
    )
    
    print('Creating DataLoaders...')
    trainloader = data_utils.DataLoader(
        train_data,
        shuffle=True,
        batch_size=paramDict['batch_size']
    )
    testloader = data_utils.DataLoader(
        test_data,
        shuffle=True,
        batch_size=paramDict['batch_size']
    )
    validloader = data_utils.DataLoader(
        valid_data,
        shuffle=True,
        batch_size=paramDict['batch_size']
    )
    
    print('Saving dataloaders...')
    with open("datasets/trainloader_descriptors.pkl", "wb") as f:
        pickle.dump(trainloader, f)
    with open("datasets/testloader_descriptors.pkl", "wb") as f:
        pickle.dump(testloader, f)
    with open("datasets/validloader_descriptors.pkl", "wb") as f:
        pickle.dump(validloader, f)
        
    return trainloader, testloader, validloader

paramDict = {
    'epoch': 100,
    'batch_size': 32,
    'dropout': 0.2,
    'loss': nn.BCELoss(),
    'input_size': 227271,
    'hidden_size': 128,
    'output_size': 2,
    'patience': 15
}

if exists('datasets/trainloader_descriptors.pkl') == False:
    print('Loading fps_x and fps_y from pickle...')
    with open("datasets/fps_x.pkl", "rb") as f:
        fps_x = pickle.load(f)
    with open("datasets/fps_y.pkl", "rb") as f:
        fps_y = pickle.load(f)
    
    trainloader, testloader, validloader = split_standardizer_dataloader(fps_x,fps_y)
else:
    print('Loading dataloaders...')
    with open("datasets/trainloader_descriptors.pkl", "rb") as f:
        trainloader = pickle.load(f)
    with open("datasets/testloader_descriptors.pkl", "rb") as f:
        testloader = pickle.load(f)
    with open("datasets/validloader_descriptors.pkl", "rb") as f:
        validloader = pickle.load(f)
    
torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0')

epochs = 100
lr = 0.004
model = MLP(
    paramDict['input_size'],
    paramDict['hidden_size'],
    paramDict['output_size'],
    paramDict['dropout']
).to(device)
optimizer = Adam(model.parameters(), lr=lr)

model = traindata(device, model, epochs, optimizer, paramDict['loss'], trainloader, validloader, paramDict['patience'])

# Test
acc, mcc, report = test(device, model, testloader)
print('Accuracy: %.3f' % acc)
print('MCC: %.3f' % mcc)
print(report)