import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
from .test import test
from .models import *
from .prepare_data import prepare_data
from ray import tune
import os


def traindata(config, device, fixed_vals, checkpoint_dir=None):
    """
    Train the model for a number of epochs.
    :param device: torch.device, the device to use.
    :param trainloader: torch.utils.data.DataLoader, the training data.
    :param validloader: torch.utils.data.DataLoader, the validation data.
    :param fixed_vals: dict, the fixed values to use.
    :param config: dict, the hyperparameters to use and tune.
    """
    
    # ------------------------------------------------------------------------------------------------
    
    trainloader, testloader, validloader, input_size = prepare_data(
        data_dir=fixed_vals['data_dir'],
        mode=fixed_vals['mode'],
        batch_size=config['batch_size'],
    )
    
    # Fixed values
    output_size = fixed_vals['output_size']
    model_label = fixed_vals['model_label']
    optimizer_label = fixed_vals['optimizer_label']
    epochs = fixed_vals['epochs']
    patience = fixed_vals['patience']
    loss_function = fixed_vals['loss_function']
    last_loss = 100
    
    # Hyperparameters to tune
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    
    if(model_label == 'mlp'):
        model = MLP(input_size, hidden_size, output_size, dropout).to(device)
    elif(model_label == 'net'):
        model = Net().to(device)
    else:
        raise ValueError("model_label must be 'mlp' or 'net'.")
    
    if(optimizer_label == 'adam'):
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif(optimizer_label == 'sgd'):
        optimizer = SGD(model.parameters(), lr=config['lr'])
    else:
        raise ValueError("optimizer_label must be either 'adam' or 'sgd'")
    
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    # ------------------------------------------------------------------------------------------------
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    # ------------------------------------------------------------------------------------------------
    
    
    for epoch in range(1, epochs+1):
        model.train()

        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            output = model(inputs)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()

            # Show progress
            if i % 100 == 0 or i == len(trainloader):
                print(f'[{epoch}/{epochs}, {i}/{len(trainloader)}] loss: {loss.item():.8}')
                
        # Early stopping
        current_loss, val_acc, val_mcc = validation(model, device, validloader, loss_function)
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
                tune.report(loss=current_loss, accuracy=val_acc, mcc=val_mcc)
                return 

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = current_loss
        scheduler.step(current_loss)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=current_loss, accuracy=val_acc, mcc=val_mcc)

def validation(model, device, validloader, loss_function):
    model.eval()
    loss_total = 0

    with torch.no_grad():
        for (inputs, targets) in validloader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = loss_function(output, targets)
            loss_total += loss.item()
            
    acc, mcc, report = test(device, model, validloader)

    return loss_total / len(validloader), acc, mcc