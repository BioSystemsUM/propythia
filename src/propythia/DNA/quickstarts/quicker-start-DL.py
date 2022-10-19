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
from utils import print_metrics, seed_everything
from src import traindata, test, DNAEncoder, data_splitting, oversample


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

def predict(model, testloader):
    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    data_dir = config['combination']['data_dir']
    kmer_one_hot = config['fixed_vals']['kmer_one_hot']
    
    metrics = test(device, model, testloader)
    print_metrics(model_label, mode, data_dir, kmer_one_hot, metrics)

if __name__ == "__main__":
    # Setting up the environment 
    config = read_config(filename='config.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    # Reading the dataset in csv format 
    reader = ReadDNA()
    filename = config['combination']['data_dir'] + '/dataset.csv'
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
        batch_size = config['hyperparameters']['batch_size']
        fps_x, fps_y = descriptors() if config['combination']['mode'] == 'descriptor' else encoding()
        fps_x, fps_y = oversample(fps_x, fps_y, config['combination']['mode'])
        trainloader, testloader, validloader = data_splitting(fps_x, fps_y, batch_size)
        model = traindata(config['hyperparameters'], device, config, trainloader, validloader)
        predict(model, testloader)
