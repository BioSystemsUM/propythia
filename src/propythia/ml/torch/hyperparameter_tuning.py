import torch
import os
from .train import traindata
from .test import test
from .models import *
from .prepare_data import prepare_data
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune import CLIReporter
from functools import partial

import sys
sys.path.append("../")
from .utils import seed_everything, print_metrics

def hyperparameter_tuning(device, config):
    """
    Hyperparameter tuning for the deep learning model.
    :param device: The device to use for the model.
    :param config: The configuration for the model.
    """

    cpus_per_trial = config['fixed_vals']['cpus_per_trial']
    gpus_per_trial = config['fixed_vals']['gpus_per_trial']
    num_samples = config['fixed_vals']['num_samples']
    epochs = config['fixed_vals']['epochs']
    model_label = config['combination']['model_label']
    data_dir = config['combination']['data_dir']
    mode = config['combination']['mode']
    class_weights = config['combination']['class_weights']
    kmer_one_hot = config['fixed_vals']['kmer_one_hot']
    output_size = config['fixed_vals']['output_size']

    seed_everything()

    # ------------------------------------------------------------------------------------------

    scheduler = HyperBandScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration", 'mcc']
    )

    result = tune.run(
        partial(
            prepare_and_train,
            device=device,
            config_from_json=config
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config['hyperparameter_search_space'],
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial('mcc', 'max', 'last')
    print("Best trial config:", best_trial.config)
    print("Best trial final validation loss:", best_trial.last_result["loss"])
    print("Best trial final validation accuracy:", best_trial.last_result["accuracy"])
    print("Best trial final validation mcc:", best_trial.last_result["mcc"])

    _, testloader, _, input_size, sequence_length = prepare_data(
        data_dir=data_dir,
        mode=mode,
        batch_size=best_trial.config['batch_size'],
        k=kmer_one_hot,
    )
    
    if model_label == 'mlp':
        best_trained_model = MLP(input_size, best_trial.config['hidden_size'], output_size, best_trial.config['dropout'])
    elif model_label == 'cnn':
        best_trained_model = CNN(input_size, best_trial.config['hidden_size'], output_size, best_trial.config['dropout'], sequence_length)
    elif model_label == 'lstm':
        best_trained_model = LSTM(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], output_size, sequence_length, best_trial.config['dropout'], device)
    elif model_label == 'bi_lstm':
        best_trained_model = LSTM(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], output_size, sequence_length, best_trial.config['dropout'], device)
    elif model_label == 'gru':
        best_trained_model = GRU(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], output_size, sequence_length, best_trial.config['dropout'], device)
    elif model_label == 'bi_gru':
        best_trained_model = GRU(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], output_size, sequence_length, best_trial.config['dropout'], device)
    elif model_label == 'cnn_lstm':
        best_trained_model = CNN_LSTM(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], sequence_length, output_size, best_trial.config['dropout'], device)
    elif model_label == 'cnn_bi_lstm':
        best_trained_model = CNN_LSTM(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], sequence_length, output_size, best_trial.config['dropout'], device)
    elif model_label == 'cnn_gru':
        best_trained_model = CNN_GRU(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], sequence_length, output_size, best_trial.config['dropout'], device)
    elif model_label == 'cnn_bi_gru':
        best_trained_model = CNN_GRU(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], sequence_length, output_size, best_trial.config['dropout'], device)
    else:
        raise ValueError('Model label not implemented', model_label)
        

    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    metrics = test(device, best_trained_model, testloader)
    print_metrics(model_label, mode, data_dir, kmer_one_hot, class_weights, metrics)

def prepare_and_train(config, device, config_from_json):
    
    data_dir = config_from_json['combination']['data_dir']
    mode = config_from_json['combination']['mode']
    kmer_one_hot = config_from_json['fixed_vals']['kmer_one_hot']
    batch_size = config['batch_size']
    
    trainloader, _, validloader, input_size, sequence_length = prepare_data(
        data_dir=data_dir,
        mode=mode,
        batch_size=batch_size,
        k=kmer_one_hot,
    )
    
    traindata(config, device, config_from_json, trainloader, validloader, input_size, sequence_length)