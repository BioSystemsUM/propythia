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
from src import prepare_data, test, hyperparameter_tuning, traindata
from ray import tune
from utils import combinations, seed_everything, calculate_possibilities

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything()

def read_config(filename='config.json'):
    """
    Reads the configuration file and validates the values. Returns the configuration.
    """
    with open(filename) as f:
        config = json.load(f)
    
    # --------------------------------------------------------------------------------------------------
    # ------------------------------------ check if data_dir exists ------------------------------------
    # --------------------------------------------------------------------------------------------------
    current_path = os.getcwd()
    current_path = current_path.replace("/quickstarts", "") # when running from notebook
    
    config['combination']['data_dir'] = current_path + '/datasets/' + config['combination']['data_dir']
    if not os.path.exists(config['combination']['data_dir']):
        raise ValueError("Data directory does not exist:", config['combination']['data_dir'])    

    # --------------------------------------------------------------------------------------------------
    # --------------------------- check if model and mode combination is valid -------------------------
    # --------------------------------------------------------------------------------------------------
    model_label = config['combination']['model_label']
    mode = config['combination']['mode']
    if(model_label in combinations):
        if(mode not in combinations[model_label]):
            raise ValueError(model_label, 'does not support', mode, ', please choose one of', combinations[model_label])
    else:
        raise ValueError('Model label:', model_label, 'not implemented in', combinations.keys())

    # --------------------------------------------------------------------------------------------------
    # --------------------------- check if it's binary classification ----------------------------------
    # --------------------------------------------------------------------------------------------------
    loss = config['fixed_vals']['loss_function']
    output_size = config['fixed_vals']['output_size']
    if(loss != "cross_entropy" or output_size != 2):
        raise ValueError(
            'Model is not binary classification, please set loss_function to cross_entropy and output_size to 2')

    # --------------------------------------------------------------------------------------------------
    # --------------------------- create the cross entropy pytorch object ------------------------------
    # --------------------------------------------------------------------------------------------------
    class_weights = torch.tensor(config['combination']['class_weights']).to(device)
    config['fixed_vals']['loss_function'] = nn.CrossEntropyLoss(weight=class_weights)
    
    # --------------------------------------------------------------------------------------------------
    # --------------------------- create ray tune objects ----------------------------------------------
    # --------------------------------------------------------------------------------------------------
    config['hyperparameter_search_space']["hidden_size"] = tune.choice(config['hyperparameter_search_space']['hidden_size'])
    config['hyperparameter_search_space']["lr"] = tune.choice(config['hyperparameter_search_space']['lr'])
    config['hyperparameter_search_space']["batch_size"] = tune.choice(config['hyperparameter_search_space']['batch_size'])
    config['hyperparameter_search_space']["dropout"] = tune.choice(config['hyperparameter_search_space']['dropout'])
    
    if config['combination']['model_label'] not in ['mlp', 'cnn']:
        config['hyperparameter_search_space']["num_layers"] = tune.choice(config['hyperparameter_search_space']['num_layers'])
    
    return config


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
        dataset_file_format = config['fixed_vals']['dataset_file_format']
        save_to_pickle = config['fixed_vals']['save_to_pickle']
        
        # get the test data
        trainloader, testloader, validloader = prepare_data(
            data_dir=data_dir,
            mode=mode,
            batch_size=batch_size,
            k=kmer_one_hot,
            dataset_file_format=dataset_file_format,
            save_to_pickle=save_to_pickle
        )
        
        # train the model
        model = traindata(hyperparameters, device, config, trainloader, validloader)
        
        # test the model
        acc, mcc, report = test(device, model, testloader)
        print("Results in test set:")
        print("--------------------")
        print("- model:        ", model_label)
        print("- mode:         ", mode)
        print("- dataset:      ", data_dir.split("/")[-1])
        print("- class weights:", class_weights)
        print("- kmer one hot: ", kmer_one_hot)
        print("--------------------")
        print('Accuracy: %.3f' % acc)
        print('MCC: %.3f' % mcc)
        print(report)
        
if __name__ == '__main__':
    config = read_config()
    
    perform(config)
