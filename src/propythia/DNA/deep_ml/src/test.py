import torch
from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

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