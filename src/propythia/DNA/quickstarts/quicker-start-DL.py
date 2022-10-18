############################################################################
### Before running: check if the config.json file has the correct values ###
############################################################################

import sys
import os
import torch
sys.path.append("../")
from read_sequence import ReadDNA
from calculate_features import calculate_and_normalize
from sklearn.preprocessing import StandardScaler
from deep_ml import read_config
from utils import seed_everything
from src import traindata, test, DNAEncoder, data_splitting


def descriptors():
    print("Calculating descriptors...")
    fps_x, fps_y = calculate_and_normalize(data)

    scaler = StandardScaler().fit(fps_x)
    fps_x = scaler.transform(fps_x)
    fps_y = fps_y.to_numpy()
    
    return fps_x, fps_y

def encoding():
    print("Calculating encodings...")
    fps_x = data['sequence'].values
    fps_y = data['label'].values
    
    encoder = DNAEncoder(fps_x)
    fps_x = encoder.one_hot_encode()
    
    return fps_x, fps_y

def data_split(batch_size, fps_x, fps_y):
    train_size = 0.6
    validation_size = 0.2
    test_size = 0.2

    trainloader, testloader, validloader = data_splitting(fps_x, fps_y, batch_size, train_size, test_size, validation_size)
    
    return trainloader, testloader, validloader

def train(config, trainloader, validloader):
    hyperparameters = config['hyperparameters']
    model = traindata(hyperparameters, device, config, trainloader, validloader)
    return model

def predict(model, testloader):
    acc, mcc, report = test(device, model, testloader)

    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    data_dir = config['combination']['data_dir']

    print("Results in test set:")
    print("--------------------")
    print("- model:  ", model_label)
    print("- mode:   ", mode)
    print("- dataset:", data_dir.split("/")[-1])
    print("--------------------")
    print('Accuracy: %.3f' % acc)
    print('MCC: %.3f' % mcc)
    print(report)

if __name__ == "__main__":
    # Setting up the environment 
    config = read_config(filename='config.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    # Reading the dataset in csv format 
    reader = ReadDNA()
    filename = "../datasets/primer/dataset.csv"
    data = reader.read_csv(filename, with_labels=True)
    print(data.head())
    print("Dataset shape:", data.shape)
    
    if config['do_tuning']:
        # this weird import was the only way I found to make it work
        os.chdir('../')
        sys.path.append(os.getcwd())
        from src import hyperparameter_tuning
        
        hyperparameter_tuning(device, config)
    else:
        fps_x, fps_y = descriptors() if config['combination']['mode'] == 'descriptor' else encoding()
        batch_size = config['hyperparameters']['batch_size']
        trainloader, testloader, validloader = data_split(batch_size, fps_x, fps_y)
        model = train(config, trainloader, validloader)
        predict(model, testloader)
