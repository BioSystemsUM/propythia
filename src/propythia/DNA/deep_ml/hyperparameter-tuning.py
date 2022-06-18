import torch
from torch import nn
import os
from src.train import traindata
from src.test import test
from src.models import MLP
from src.prepare_data import prepare_data
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial

# ------------------------------------------------------------------------------------------------

torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------

class_weights = torch.tensor([1.0, 1.0]).to(device)
# class_weights = torch.tensor([1.0, 4.0]).to(device)

fixed_vals = {
    'epochs': 50,
    'optimizer_label': 'adam',
    'loss_function': nn.CrossEntropyLoss(weight=class_weights),
    'patience': 2, 
    'output_size': 2,     
    'model_label': 'mlp',
    'data_dir': os.path.abspath('datasets/primer'),
    'mode': 'descriptors' # 'descriptors' or 'one hot encode'
}

# these are hyperparameters to be tuned
config = {
    "hidden_size": tune.choice([32, 64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8, 16, 32]),
    "dropout": tune.uniform(0.3, 0.5)
}

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

# ------------------
cpus_per_trial = 2
gpus_per_trial = 2
num_samples = 15
# ------------------

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

# best trial is to maximize the mcc
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
best_trained_model = MLP(input_size, best_trial.config['hidden_size'], fixed_vals['output_size'], best_trial.config['dropout'])
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

# ----------------------------------------------------------------------


# model = traindata(device, trainloader, validloader, fixed_vals, config)

# # Test
# acc, mcc, report = test(device, model, testloader)
# print('Accuracy: %.3f' % acc)
# print('MCC: %.3f' % mcc)
# print(report)
