import sys
sys.path.append("../")
import os
import torch
import torch.utils.data as data_utils
import pickle
from propythia.dna.calculate_features import calculate_and_normalize
from propythia.dna.read_sequence import ReadDNA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from propythia.dna.encoding import DNAEncoder

from .utils import seed_everything

def data_splitting(fps_x, fps_y, batch_size, train_size, test_size, validation_size):
    """
    Split data into train, test and validation sets.
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
    
    return trainloader, testloader, validloader, x_train

def prepare_data(data_dir, mode, batch_size, k, train_size=0.6, test_size=0.2, validation_size=0.2):
    """
    Prepare data for training and testing.
    :param data_dir: str, the path to the data directory.
    :param mode: str, the mode to use. Must be either 'descriptor', 'one_hot', 'chemical' or 'kmer_one_hot'.
    :param batch_size: int, the batch size to use.
    :param k: int, value for the kmer one hot encoding.
    :param train_size: float, the proportion of the data to use for training.
    :param test_size: float, the proportion of the data to use for testing.
    :param validation_size: float, the proportion of the data to use for validation.
    :return: trainloader: torch.utils.data.DataLoader, the training data.
    :return: testloader: torch.utils.data.DataLoader, the testing data.
    :return: validloader: torch.utils.data.DataLoader, the validation data.
    :return: input_size: int, the size of the input.
    :return: sequence_length: int, the size of the length of sequence.
    """
    
    seed_everything()
    
    fps_x_file = data_dir + '/fps_x_descriptor.pkl' if mode == 'descriptor' else data_dir + '/fps_x.pkl'
    fps_y_file = data_dir + '/fps_y_descriptor.pkl' if mode == 'descriptor' else data_dir + '/fps_y.pkl'
    
    # check if fps_x_file and fps_y_file exist
    if not os.path.isfile(fps_x_file) or not os.path.isfile(fps_y_file):
        
        # read data from csv
        reader = ReadDNA()
        data = reader.read_csv(filename=data_dir + '/dataset.csv', with_labels=True)
        
        # calculate features
        if mode == 'descriptor':
            fps_x, fps_y = calculate_and_normalize(data)
        else:
            fps_x = data['sequence'].values
            fps_y = data['label'].values
        
        # save fps_x and fps_y to files
        with open(fps_x_file, 'wb') as f:
            pickle.dump(fps_x, f)
        with open(fps_y_file, 'wb') as f:
            pickle.dump(fps_y, f)
    else:
        # load fps_x and fps_y from files
        with open(fps_x_file, 'rb') as f:
            fps_x = pickle.load(f)
        with open(fps_y_file, 'rb') as f:
            fps_y = pickle.load(f)
    
    if(mode in ['one_hot', 'chemical', 'kmer_one_hot']):
        encoder = DNAEncoder(fps_x)
        possibilities = {
            'one_hot': encoder.one_hot_encode,
            'chemical': encoder.chemical_encode,
            'kmer_one_hot': encoder.kmer_one_hot_encode
        }
        fps_x = possibilities[mode]() if mode != 'kmer_one_hot' else possibilities[mode](k)
    elif(mode == 'descriptor'):
        scaler = StandardScaler().fit(fps_x)
        fps_x = scaler.transform(fps_x)
        fps_y = fps_y.to_numpy()
    else:
        raise ValueError("mode must be either 'one_hot', 'descriptor', 'chemical' or 'kmer_one_hot_encode'.")
    
    trainloader, testloader, validloader, x_train = data_splitting(fps_x, fps_y, batch_size, train_size, test_size, validation_size)
    
    return trainloader, testloader, validloader, x_train.shape[-1], x_train.shape[1]
