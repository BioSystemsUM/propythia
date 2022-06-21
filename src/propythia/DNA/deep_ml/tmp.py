import pickle
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils
from src_old.prepare_data import prepare_data
from src.encoding import DNAEncoding
from src.models import RNN
import torch.optim as optim
from src.test import test
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.autograd import Variable
import os

# load fps_x and fps_y from essential genes pickle files
with open('datasets/primer/fps_x.pkl', 'rb') as f:
    fps_x = pickle.load(f)
with open('datasets/primer/fps_y.pkl', 'rb') as f:
    fps_y = pickle.load(f)

# ------------------------------------------------------------------------------------------

def traindata(device, model, epochs, optimizer, loss_function, train_loader, valid_loader, patience):
    
    # Early stopping
    last_loss = 100
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(1, epochs+1):
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            output = model(inputs)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()

            # Show progress
            if i % 100 == 0 or i == len(train_loader):
                print(f'[{epoch}/{epochs}, {i}/{len(train_loader)}] loss: {loss.item():.8}')
                
        # Early stopping
        current_loss = validation(model, device, valid_loader, loss_function)
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = current_loss
        scheduler.step(current_loss)

    return model

def validation(model, device, valid_loader, loss_function):
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for (inputs, targets) in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = loss_function(output, targets)
            loss_total += loss.item()

    return loss_total / len(valid_loader)

# ------------------------------------------------------------------------------------------

mode = 'one_hot'

torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_hidden = 128
input_size = 4
n_categories = 2
model = RNN(input_size, n_hidden, n_categories)

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss()

trainloader, testloader, validloader = prepare_data(
    fps_x, fps_y, 
    mode=mode,
    batch_size=32,
)

optimizer = Adam(model.parameters(), lr=0.001)

model = traindata(device, model, 50, optimizer , criterion, trainloader, validloader, 2)

# Test
acc, mcc, report = test(device, model, testloader)
print('Accuracy: %.3f' % acc)
print('MCC: %.3f' % mcc)
print(report)

def train(category_tensor, line_tensor):
    hidden = model.initHidden()

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

