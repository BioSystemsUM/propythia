import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from numpy import vstack
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from numpy import argmax

import encoding as enc


# Model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 20, 10, stride=1, padding=0)
        self.fc1 = nn.Linear(940, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)

        self.max_pool = nn.MaxPool1d(10, stride=5)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x.shape: ", x.shape)
        # print("x: ", x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act3(x)

        return x


# Train
def traindata(device, model, epochs, optimizer, loss_function, train_loader, valid_loader):
    # Early stopping
    last_loss = 100
    patience = 2

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


def test(device, model, test_loader):

    model.eval()

    predictions, actuals = list(), list()
    with torch.no_grad():
        for (inputs, targets) in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            yhat = model(inputs)
            yhat = yhat.cpu().detach().numpy()
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), 1))
            yhat = argmax(yhat, axis=1)
            yhat = yhat.reshape((len(yhat), 1))
            predictions.append(yhat)
            actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    mcc = matthews_corrcoef(actuals, predictions)
    report = confusion_matrix(actuals, predictions)
    return acc, mcc, report


def main():
    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    epochs = 100
    batch_size = 16
    lr = 0.004
    loss_function = nn.CrossEntropyLoss()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = pd.read_csv("datasets/human-exercise/dataset.csv")
    print("dataset.shape:", dataset.shape)

    fps_x = dataset['sequence'].values
    fps_y = dataset['label'].values

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

    print("Encoding...")
    x_train_enc = enc.DNAEncoding(x_train)
    x_train = x_train_enc.one_hot_encode()

    x_test_enc = enc.DNAEncoding(x_test)
    x_test = x_test_enc.one_hot_encode()

    x_cv_enc = enc.DNAEncoding(x_cv)
    x_cv = x_cv_enc.one_hot_encode()

    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    print("x_cv.shape:", x_cv.shape)
    print("y_cv.shape:", y_cv.shape)

    # convert to torch.tensor
    train_data = data_utils.TensorDataset(
        torch.tensor(x_train, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_data = data_utils.TensorDataset(
        torch.tensor(x_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.long)
    )
    valid_data = data_utils.TensorDataset(
        torch.tensor(x_cv, dtype=torch.float),
        torch.tensor(y_cv, dtype=torch.long)
    )

    # Data loader
    trainloader = data_utils.DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size
    )
    testloader = data_utils.DataLoader(
        test_data,
        shuffle=True,
        batch_size=batch_size
    )
    validloader = data_utils.DataLoader(
        valid_data,
        shuffle=True,
        batch_size=batch_size
    )

    # Train
    model = traindata(device, model, epochs, optimizer, loss_function, trainloader, validloader)

    # Test
    acc, mcc, report = test(device, model, testloader)
    print('Accuracy: %.3f' % acc)
    print('MCC: %.3f' % mcc)
    print(report)


if __name__ == '__main__':
    main()
