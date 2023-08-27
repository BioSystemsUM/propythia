"""
########################################################################
Runs a combination of hyperparameters or performs hyperparameter tuning
for the given model, feature mode, and data directory.
########################################################################
"""

import torch
import os
from .prepare_data import prepare_data
from .test import test
from .hyperparameter_tuning import hyperparameter_tuning
from .train import traindata
from .utils import print_metrics, read_config

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def perform(config):
    if config['do_tuning']:
        hyperparameter_tuning(device, config)
    else:
        model_label = config['combination']['model_label']
        mode = config['combination']['mode']
        data_dir = config['combination']['data_dir']
        class_weights = config['combination']['class_weights']
        batch_size = config['hyperparameters']['batch_size']
        kmer_one_hot = config['fixed_vals']['kmer_one_hot']
        hyperparameters = config['hyperparameters']
        
        trainloader, testloader, validloader, input_size, sequence_length = prepare_data(
            data_dir=data_dir,
            mode=mode,
            batch_size=batch_size,
            k=kmer_one_hot,
        )
        
        # train the model
        model = traindata(hyperparameters, device, config, trainloader, validloader, input_size, sequence_length)

        # test the model
        metrics = test(device, model, testloader)
        print_metrics(model_label, mode, data_dir, kmer_one_hot, class_weights, metrics)
        
if __name__ == '__main__':
    config = read_config(device)
    perform(config)