from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .encoding import DNAEncoding
import torch
import torch.utils.data as data_utils
import pickle

def prepare_data(data_dir, mode, batch_size, train_size=0.6, test_size=0.2, validation_size=0.2):
    """
    Prepare data for training and testing.
    :param data_dir: str, the path to the data directory.
    :param mode: str, the mode to use. Must be either 'descriptor' or 'one_hot'.
    :param batch_size: int, the batch size to use.
    :param train_size: float, the proportion of the data to use for training.
    :param test_size: float, the proportion of the data to use for testing.
    :param validation_size: float, the proportion of the data to use for validation.
    :return: trainloader: torch.utils.data.DataLoader, the training data.
    :return: testloader: torch.utils.data.DataLoader, the testing data.
    :return: validloader: torch.utils.data.DataLoader, the validation data.
    :return: input_size: int, the size of the input.
    """
    
    try:
        with open(data_dir + '/fps_x.pkl', 'rb') as f:
            fps_x = pickle.load(f)
        with open(data_dir + '/fps_y.pkl', 'rb') as f:
            fps_y = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find fps_x.pkl and fps_y.pkl in" + data_dir)
    
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
    return trainloader, testloader, validloader, x_train.shape[-1]