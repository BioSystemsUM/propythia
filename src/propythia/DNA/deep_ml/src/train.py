import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

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