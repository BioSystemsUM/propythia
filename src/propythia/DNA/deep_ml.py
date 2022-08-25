"""
########################################################################
Runs a combination of hyperparameters or performs hyperparameter tuning
for the given model, feature mode, and data directory.
########################################################################
"""

import json
import torch
from torch import nn
import os
from src.prepare_data import prepare_data
from src.test import test
from src.hyperparameter_tuning import hyperparameter_tuning
from src.train import traindata
from ray import tune
import numpy
from utils import combinations

numpy.random.seed(2022)
torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)

def create_objects_from_config(config):
    # --------- create the cross entropy pytorch object ---------
    class_weights = torch.tensor([1.0, 1.0]).to(device)
    config['fixed_vals']['loss_function'] = nn.CrossEntropyLoss(weight=class_weights)
    
    # --------- create ray tune objects ---------
    config['hyperparameter_search_space'] = {
        "hidden_size": tune.choice(config['hyperparameter_search_space']['hidden_size']),
        "lr": tune.loguniform(config['hyperparameter_search_space']['lr'][0], config['hyperparameter_search_space']['lr'][1]),
        "batch_size": tune.choice(config['hyperparameter_search_space']['batch_size']),
        "dropout": tune.uniform(config['hyperparameter_search_space']['dropout'][0], config['hyperparameter_search_space']['dropout'][1]),
        "num_layers": tune.choice(config['hyperparameter_search_space']['num_layers']),
    }
    return config

def read_config(filename='config.json'):
    """
    Reads the configuration file and validates the values. Returns the configuration.
    """
    with open(filename) as f:
        config = json.load(f)
    
    current_path = os.getcwd()
    current_path = current_path.replace("/notebooks", "") # when running from notebook
    
    # --------- check if data_dir exists ---------
    config['combination']['data_dir'] = current_path + '/datasets/' + config['combination']['data_dir']
    if not os.path.exists(config['combination']['data_dir']):
        raise ValueError("Data directory does not exist:", config['combination']['data_dir'])    

    # --------- check if model and mode combination is valid ---------
    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    if(model_label in combinations):
        if(mode not in combinations[model_label]):
            raise ValueError(model_label, 'does not support', mode, ', please choose one of', combinations[model_label])
    else:
        raise ValueError('Model label:', model_label, 'not implemented in', combinations.keys())

    # --------- check if it's binary classification ---------
    loss = config['fixed_vals']['loss_function']
    output_size = config['fixed_vals']['output_size']
    if(loss != "cross_entropy" or output_size != 2):
        raise ValueError(
            'Model is not binary classification, please set loss_function to cross_entropy and output_size to 2')

    # --------- check if hyperparameters search space is valid ---------
    if len(config['hyperparameter_search_space']['lr']) != 2:
        raise ValueError('lr must be a list of length 2 and a random value between the two values will be chosen')
    if len(config['hyperparameter_search_space']['dropout']) != 2:
        raise ValueError('dropout must be a list of length 2 and a random value between the two values will be chosen')

    config = create_objects_from_config(config)

    return config


def perform(config):
    
    if config['do_tuning']:
        hyperparameter_tuning(device, config)
    else:
        model_label = config['combination']['model_label']
        mode = config['combination']['mode']
        data_dir = config['combination']['data_dir']
        batch_size = config['hyperparameters']['batch_size']
        kmer_one_hot = config['fixed_vals']['kmer_one_hot']
        hyperparameters = config['hyperparameters']
        
        # train the model
        model = traindata(hyperparameters, device, config)
        
        # get the test data
        _, testloader, _, _, _ = prepare_data(
            data_dir=data_dir,
            mode=mode,
            batch_size=batch_size,
            k=kmer_one_hot,
        )

        # test the model
        acc, mcc, report = test(device, model, testloader)
        print("Results in test set:")
        print("--------------------")
        print("- model:  ", model_label)
        print("- mode:   ", mode)
        print("- dataset:", data_dir.split("/")[-1])
        print("--------------------")
        print('Accuracy: %.3f' % acc)
        print('MCC: %.3f' % mcc)
        print(report)
        
if __name__ == '__main__':
    config = read_config()
    print("Starting training:", config['combination']['model_label'], config['combination']['mode'], config['combination']['data_dir'])
    if config['train_all_combinations']:
        for model_label in combinations:
            for mode in combinations[model_label]:
                print("Training model:", model_label, "mode:", mode)
                config['combination']['model_label'] = model_label
                config['combination']['mode'] = mode
                perform(config)
    else:
        perform(config)
