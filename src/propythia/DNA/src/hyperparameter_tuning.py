import torch
import os
from .train import traindata
from .test import test
from .models import *
from .prepare_data import prepare_data
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial


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
    kmer_one_hot = config['fixed_vals']['kmer_one_hot']
    output_size = config['fixed_vals']['output_size']

    # ------------------------------------------------------------------------------------------

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration", 'mcc']
    )

    result = tune.run(
        partial(
            traindata,
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

    models = {
        'mlp': MLP(input_size, best_trial.config['hidden_size'], output_size, best_trial.config['num_layers'], best_trial.config['dropout']),
        'cnn': CNN(sequence_length, input_size, best_trial.config['hidden_size'], output_size),
        'lstm': LSTM(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], output_size, sequence_length, device),
        'bi_lstm': LSTM(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], output_size, sequence_length, device),
        'gru': GRU(input_size, best_trial.config['hidden_size'], best_trial.config['num_layers'], output_size, sequence_length, device),
        'cnn_lstm': CNN_LSTM(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], sequence_length, output_size, device),
        'cnn_bi_lstm': CNN_LSTM(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], sequence_length, output_size, device),
        'cnn_gru': CNN_GRU(input_size, best_trial.config['hidden_size'], False, best_trial.config['num_layers'], sequence_length, output_size, device),
        'cnn_bi_gru': CNN_GRU(input_size, best_trial.config['hidden_size'], True, best_trial.config['num_layers'], sequence_length, output_size, device)
    }

    if(model_label in models):
        best_trained_model = models[model_label]
    else:
        raise ValueError(
            'Model label not implemented', model_label,
            'only implemented models are', models.keys())

    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    acc, mcc, report = test(device, best_trained_model, testloader)
    print("Results in test set:")
    print("--------------------")
    print("- model:  ", model_label)
    print("- mode:   ", mode)
    print("- dataset:", data_dir.split("/")[-1])
    print("--------------------")
    print('Accuracy: %.3f' % acc)
    print('MCC: %.3f' % mcc)
    print(report)
