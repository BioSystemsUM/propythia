import torch
from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, recall_score, roc_auc_score


def test(device, model, test_loader):
    """
    Test the model.
    :param model: Model to be tested.
    :param device: Device to be used for testing.
    :param test_loader: Data loader for testing.
    :return: The accuracy, mcc and confusion matrix of the model.
    """
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
    roc_auc = roc_auc_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    report = confusion_matrix(actuals, predictions)

    metrics = {
        'accuracy': acc,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'f1': f1,
        'recall': recall,
        'confusion_matrix': report
    }
    return metrics
