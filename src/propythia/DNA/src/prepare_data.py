import os
import torch
import torch.utils.data as data_utils
import pickle
from calculate_features import calculate_and_normalize
from read_sequence import ReadDNA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .encoding import DNAEncoder
import sys
sys.path.append("../")
from utils import seed_everything, save_pickle, load_pickle

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
    
    return trainloader, testloader, validloader

def calculate_descriptors(data):
    fps_x, fps_y = calculate_and_normalize(data)
    scaler = StandardScaler().fit(fps_x)
    fps_x = scaler.transform(fps_x)
    fps_y = fps_y.to_numpy()
    
    return fps_x, fps_y

def calculate_encoders(data, mode, k):
    fps_x = data['sequence'].values
    fps_y = data['label'].values
    
    encoder = DNAEncoder(fps_x)
    if mode == 'one_hot':
        fps_x = encoder.one_hot_encode()
    if mode == 'chemical':
        fps_x = encoder.chemical_encode()
    if mode == 'kmer_one_hot':
        fps_x = encoder.kmer_one_hot_encode(k)
        
    return fps_x, fps_y

def prepare_data(data_dir, mode, batch_size, k, dataset_file_format, save_to_pickle = True, train_size=0.6, test_size=0.2, validation_size=0.2):
    """
    Prepare data for training and testing.
    :param data_dir: str, the path to the data directory.
    :param mode: str, the mode to use. Must be either 'descriptor', 'one_hot', 'chemical' or 'kmer_one_hot'.
    :param batch_size: int, the batch size to use.
    :param k: int, value for the kmer one hot encoding.
    :param dataset_file_format: str, the file format of the dataset. Must be either 'csv' or 'fasta'.
    :param save_to_pickle: bool, whether to save the data to pickle files.
    :param train_size: float, the proportion of the data to use for training.
    :param test_size: float, the proportion of the data to use for testing.
    :param validation_size: float, the proportion of the data to use for validation.
    :return: trainloader: torch.utils.data.DataLoader, the training data.
    :return: testloader: torch.utils.data.DataLoader, the testing data.
    :return: validloader: torch.utils.data.DataLoader, the validation data.
    """
    
    seed_everything()
    
    if mode == 'descriptor':
        fps_x_file = data_dir + '/fps_x_descriptor.pkl'
        fps_y_file = data_dir + '/fps_y_descriptor.pkl'
    if mode == 'one_hot' or mode == 'chemical':
        fps_x_file = data_dir + '/fps_x_' + mode + '.pkl'
        fps_y_file = data_dir + '/fps_y_' + mode + '.pkl'
    if mode == 'kmer_one_hot':
        fps_x_file = data_dir + '/fps_x_' + mode + '_' + str(k) + '.pkl'
        fps_y_file = data_dir + '/fps_y_' + mode + '_' + str(k) + '.pkl'
    
    # check if fps_x_file and fps_y_file exist
    if not os.path.isfile(fps_x_file) or not os.path.isfile(fps_y_file):
        print("Calculating features...")
        
        # read data from csv
        reader = ReadDNA()
        if dataset_file_format == 'csv':
            data = reader.read_csv(filename=data_dir + '/dataset.csv', with_labels=True)
        elif dataset_file_format == 'fasta':
            data = reader.read_fasta(filename=data_dir + '/dataset.fasta', with_labels=True)
        else:
            raise ValueError("dataset_file_format must be either 'csv' or 'fasta'.")
        
        # calculate features
        fps_x, fps_y = calculate_descriptors(data) if mode == 'descriptor' else calculate_encoders(data, mode, k) 
        
        if save_to_pickle:
            save_pickle(fps_x, fps_x_file)
        
    else:
        print("Loading data from files...")
        fps_x, fps_y = load_pickle(fps_x_file, fps_y_file)
    
    trainloader, testloader, validloader = data_splitting(fps_x, fps_y, batch_size, train_size, test_size, validation_size)
    
    return trainloader, testloader, validloader
