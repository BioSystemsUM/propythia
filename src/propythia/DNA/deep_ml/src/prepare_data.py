from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .encoding import DNAEncoding
import torch
import torch.utils.data as data_utils
import torch.nn as nn

def prepare_data(fps_x, fps_y, mode, batch_size, train_size=0.6, test_size=0.2, validation_size=0.2):
    """
    Prepare data for training and testing.
    :param fps_x: list of file paths of x data. 
    :param fps_y: list of file paths of y data.
    :param mode: str, 'one_hot' or 'descriptors'.
    :param batch_size: int, batch size.
    :param train_size: float, the proportion of training data.
    :param test_size: float, the proportion of testing data.
    :param validation_size: float, the proportion of validation data.
    
    :return:
    """
    
    if(train_size + test_size + validation_size != 1):
        raise ValueError("The sum of train_size, test_size and validation_size must be 1.")
    
    x, x_test, y, y_test = train_test_split(
        fps_x, fps_y,
        test_size=test_size,
        train_size=train_size + validation_size,
        stratify=fps_y
    )
    x_train, x_cv, y_train, y_cv = train_test_split(
        x, y,
        test_size=validation_size/(1-test_size),
        train_size=1-(validation_size/(1-test_size)),
        stratify=y
    )
    
    if(mode == 'one_hot'):
        encoder = DNAEncoding(x_train)
        x_train = encoder.one_hot_encode()

        encoder = DNAEncoding(x_test)
        x_test = encoder.one_hot_encode()

        encoder = DNAEncoding(x_cv)
        x_cv = encoder.one_hot_encode()
    else:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_cv = scaler.transform(x_cv)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        y_cv = y_cv.to_numpy()

    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    print("x_cv.shape:", x_cv.shape)
    print("y_cv.shape:", y_cv.shape)
    
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
    
    return trainloader, testloader, validloader