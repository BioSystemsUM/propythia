import torch
import os
from .train import traindata
from .test import test
from .models import MLP, Net
from .prepare_data import prepare_data
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial


def hyperparameter_tuning(device, fixed_vals, config):
    """
    Hyperparameter tuning for the deep learning model.
    :param device: The device to use for the model.
    :param fixed_vals: The fixed values for the model.
    :param config: The configuration for the model.
    """

    cpus_per_trial = 2
    gpus_per_trial = 2
    num_samples = 15
    fixed_vals['data_dir'] = os.path.abspath('datasets/' + fixed_vals['data_dir'])
    
    # ------------------------------------------------------------------------------------------

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=fixed_vals['epochs'],
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
            fixed_vals=fixed_vals,
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial('mcc', 'max', 'last')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation mcc: {}".format(
        best_trial.last_result["mcc"]))

    _, testloader, _, input_size = prepare_data(
        data_dir=fixed_vals['data_dir'],
        mode=fixed_vals['mode'],
        batch_size=best_trial.config['batch_size'],
    )
    if(fixed_vals['model_label'] == 'mlp'):
        best_trained_model = MLP(input_size, best_trial.config['hidden_size'], fixed_vals['output_size'], best_trial.config['dropout'])
    elif(fixed_vals['model_label'] == 'net'):
        best_trained_model = Net(input_size, best_trial.config['hidden_size'], fixed_vals['output_size'], best_trial.config['dropout'])
    else:
        raise ValueError("model_label must be 'mlp' or 'net'.")
        
        
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    acc, mcc, report = test(device, best_trained_model, testloader)
    print("Results in test set:")
    print("--------------------")
    print('Accuracy: %.3f' % acc)
    print('MCC: %.3f' % mcc)
    print(report)
