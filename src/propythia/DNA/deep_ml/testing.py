import torch
from torch import nn
import os
from src.hyperparameter_tuning import hyperparameter_tuning
from ray import tune


torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def perform(model_label, mode, data_dir):
    if('essential_genes' in data_dir):
        class_weights = torch.tensor([1.0, 2.0]).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(device)

    combinations = {
        'mlp': ['descriptor'],
        'net': ['one_hot', 'chemical'],
        'cnn': ['one_hot', 'chemical'],
        'rnn_lstm': ['one_hot', 'chemical']
    }
    if(model_label in combinations):
        if(mode not in combinations[model_label]):
            raise ValueError(model_label, 'does not support', mode, ', please choose one of', combinations[model_label])
    else:
        raise ValueError('Model label not implemented', model_label)

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

# perform('mlp', 'descriptor', 'essential_genes/descriptors_all_small_seqs')
# perform('mlp', 'descriptor', 'essential_genes/descriptors_filtered_20k')
# perform('mlp', 'descriptor', 'essential_genes/descriptors_filtered_50k')
# perform('mlp', 'descriptor', 'primer')
# perform('mlp', 'descriptor', 'essential_genes')
# perform('cnn', 'one_hot', 'primer')
# perform('cnn', 'one_hot', 'essential_genes')
# perform('cnn', 'chemical', 'primer')
# perform('cnn', 'chemical', 'essential_genes')
# perform('rnn_lstm', 'one_hot', 'primer')
# perform('rnn_lstm', 'one_hot', 'essential_genes')
perform('rnn_lstm', 'chemical', 'primer')
# perform('rnn_lstm', 'chemical', 'essential_genes')
