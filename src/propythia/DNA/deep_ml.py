"""
########################################################################
Runs a combination of hyperparameters or performs hyperparameter tuning
for the given model, feature mode, and data directory.
########################################################################
"""
import torch
import os
from src import prepare_data, test, hyperparameter_tuning, traindata
from utils import read_config, seed_everything, print_metrics

def perform(config):
    # Setting up the environment     
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything()
    
    # Defining all the parameters
    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    data_dir = config['combination']['data_dir']
    batch_size = config['hyperparameters']['batch_size']
    kmer_one_hot = config['fixed_vals']['kmer_one_hot']
    hyperparameters = config['hyperparameters']
    dataset_file_format = config['fixed_vals']['dataset_file_format']
    save_to_pickle = config['fixed_vals']['save_to_pickle']
    cutting_length = config['fixed_vals']['cutting_length']
    read_from_pickle = config['fixed_vals']['read_from_pickle']
    
    if config['do_tuning']:
        hyperparameter_tuning(device, config)
    else:
        # get the data
        trainloader, testloader, validloader = prepare_data(
            data_dir=data_dir,
            mode=mode,
            batch_size=batch_size,
            k=kmer_one_hot,
            dataset_file_format=dataset_file_format,
            cutting_length=cutting_length,
            save_to_pickle=save_to_pickle,
            read_from_pickle=read_from_pickle
        )
        
        # train the model
        model = traindata(hyperparameters, device, config, trainloader, validloader)
        
        # test the model
        metrics = test(device, model, testloader)
        print_metrics(model_label, mode, data_dir, kmer_one_hot, metrics)
        
if __name__ == '__main__':
    config = read_config()
    perform(config)
