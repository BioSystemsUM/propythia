import torch
from torch import nn
import os
from src.hyperparameter_tuning import hyperparameter_tuning
from ray import tune


torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def perform(model_label, mode, data_dir):
    if(data_dir == 'essential_genes'):
        class_weights = torch.tensor([1.0, 11.0]).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(device)

    print("class_weights: ", class_weights)
    if(model_label == 'mlp' and mode != 'descriptor'):
        raise ValueError('MLP model can only be used for descriptor mode.')
    elif(model_label == 'net' and mode == 'descriptor'):
        raise ValueError('Net model can only be used for encoding mode.')
    elif(model_label == 'cnn' and mode == 'descriptor'):
        raise ValueError('CNN model can only be used for encoding mode.')
    elif(model_label == 'rnn' and mode == 'descriptor'):
        raise ValueError('RNN model can only be used for encoding mode.')

    fixed_vals = {
        'epochs': 50,
        'optimizer_label': 'adam',
        'loss_function': nn.CrossEntropyLoss(weight=class_weights),
        'patience': 2, 
        'output_size': 2,     
        'model_label': model_label,
        'data_dir': data_dir,
        'mode': mode
    }

    config = {
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32]),
        "dropout": tune.uniform(0.3, 0.5)
    }

    hyperparameter_tuning(device, fixed_vals, config)
    

# perform('mlp', 'descriptor', 'primer')
# perform('mlp', 'descriptor', 'essential_genes')
# perform('cnn', 'one_hot', 'primer')
perform('cnn', 'one_hot', 'essential_genes')
# perform('cnn', 'chemical', 'primer')
# perform('cnn', 'chemical', 'essential_genes')
# perform('rnn', 'one_hot', 'primer')
# perform('rnn', 'one_hot', 'essential_genes')
# perform('rnn', 'chemical', 'primer')
# perform('rnn', 'chemical', 'essential_genes')