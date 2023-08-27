import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
from .test import test
from .models import *
from .prepare_data import prepare_data
from ray import tune
import os

import sys
sys.path.append("../")
from .utils import seed_everything


def traindata(config, device, config_from_json, trainloader, validloader, input_size, sequence_length, checkpoint_dir=None):
    """
    Train the model for a number of epochs.
    :param config: Dictionary of hyperparameters to be tuned.
    :param device: Device to be used for training.
    :param fixed_vals: Dictionary of fixed parameters.
    :param checkpoint_dir: Directory to save the model.
    """
    
    seed_everything()
    
    # Fixed values
    do_tuning = config_from_json['do_tuning']
    model_label = config_from_json['combination']['model_label']
    output_size = config_from_json['fixed_vals']['output_size']
    optimizer_label = config_from_json['fixed_vals']['optimizer_label']
    epochs = config_from_json['fixed_vals']['epochs']
    patience = config_from_json['fixed_vals']['patience']
    loss_function = config_from_json['fixed_vals']['loss_function']
    last_loss = 100

    # Hyperparameters to tune
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    lr = config['lr']
    num_layers = config['num_layers']

    if model_label == 'mlp':
        model = MLP(input_size, hidden_size, output_size, dropout).to(device)
    elif model_label == 'cnn':
        model = CNN(input_size, hidden_size, output_size, dropout, sequence_length).to(device)
    elif model_label == 'lstm':
        model = LSTM(input_size, hidden_size, False, num_layers, output_size, sequence_length, dropout, device).to(device)
    elif model_label == 'bi_lstm':
        model = LSTM(input_size, hidden_size, True, num_layers, output_size, sequence_length, dropout, device).to(device)
    elif model_label == 'gru':
        model = GRU(input_size, hidden_size, False, num_layers, output_size, sequence_length, dropout, device).to(device)
    elif model_label == 'bi_gru':
        model = GRU(input_size, hidden_size, True, num_layers, output_size, sequence_length, dropout, device).to(device)
    elif model_label == 'cnn_lstm':
        model = CNN_LSTM(input_size, hidden_size, False, num_layers, sequence_length, output_size, dropout, device).to(device)
    elif model_label == 'cnn_bi_lstm':
        model = CNN_LSTM(input_size, hidden_size, True, num_layers, sequence_length, output_size, dropout, device).to(device)
    elif model_label == 'cnn_gru':
        model = CNN_GRU(input_size, hidden_size, False, num_layers, sequence_length, output_size, dropout, device).to(device)
    elif model_label == 'cnn_bi_gru':
        model = CNN_GRU(input_size, hidden_size, True, num_layers, sequence_length, output_size, dropout, device).to(device)
    else:
        raise ValueError('Model label not implemented', model_label)

    if(optimizer_label == 'adam'):
        optimizer = Adam(model.parameters(), lr=lr)
    elif(optimizer_label == 'sgd'):
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_label must be either 'adam' or 'sgd'")

    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # ------------------------------------------------------------------------------------------------

    if do_tuning and checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # ------------------------------------------------------------------------------------------------

    trigger_times = 0
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

        if current_loss >= last_loss or torch.isnan(loss):
            trigger_times += 1
            if torch.isnan(loss):
                print('NAN loss! trigger_times:', trigger_times)
            else:
                print('trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                if do_tuning:
                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save((model.state_dict(), optimizer.state_dict()), path)
                    tune.report(loss=current_loss, accuracy=val_acc, mcc=val_mcc)
                    return
                else:
                    return model

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = current_loss
        scheduler.step(current_loss)
        if do_tuning:
            tune.report(loss=current_loss, accuracy=val_acc, mcc=val_mcc)
        
    # if do_tuning == False:
    return model
        

def validation(model, device, validloader, loss_function):
    """
    Validate the model.
    :param model: Model to be validated.
    :param device: Device to be used for validation.
    :param validloader: Data loader for validation.
    :param loss_function: Loss function to be used.
    :return: The loss, accuracy and mcc of the model.
    """
    model.eval()
    loss_total = 0

    with torch.no_grad():
        for (inputs, targets) in validloader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = loss_function(output, targets)
            loss_total += loss.item()

    metrics = test(device, model, validloader)

    return loss_total / len(validloader), metrics['accuracy'], metrics['mcc']
