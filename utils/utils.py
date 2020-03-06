import torch
import numpy as np
import os
import json
from models.base import Normalizer1D



# get the number of model parameters
def get_n_params(model_to_eval):

    return sum(p.numel() for p in model_to_eval.parameters() if p.requires_grad)


# compute the normalizers
def compute_normalizer(loader_train):
    # definition
    variance_scaler = 1

    # initialization
    total_batches = 0
    u_mean = 0
    y_mean = 0
    u_var = 0
    y_var = 0
    for i, (u, y) in enumerate(loader_train):
        total_batches += u.size()[0]
        u_mean += torch.mean(u, dim=(0, 2))
        y_mean += torch.mean(y, dim=(0, 2))
        u_var += torch.mean(torch.var(u, dim=2, unbiased=False), dim=(0,))
        y_var += torch.mean(torch.var(y, dim=2, unbiased=False), dim=(0,))

    u_mean = u_mean.numpy()
    y_mean = y_mean.numpy()
    u_var = u_var.numpy()
    y_var = y_var.numpy()

    """u_normalizer = (torch.tensor(np.sqrt(u_var / total_batches) * variance_scaler),
                    torch.tensor(u_mean / total_batches))
    y_normalizer = (torch.tensor(np.sqrt(y_var / total_batches) * variance_scaler),
                    torch.tensor(y_mean / total_batches))"""

    u_normalizer = Normalizer1D(np.sqrt(u_var / total_batches) * variance_scaler, u_mean / total_batches)
    y_normalizer = Normalizer1D(np.sqrt(y_var / total_batches) * variance_scaler, y_mean / total_batches)

    return u_normalizer, y_normalizer


# save the options
def save_options(options_in, path, file_name):
    # copy options without reference to old object
    options = dict(options_in)

    # replace options['device]'
    if 'device' in options:
        options['device'] = options['device'].type

    # namespaces to dictionary
    if 'dataset_options' in options:
        options['dataset_options'] = vars(options['dataset_options'])
    if 'model_options' in options:
        options['model_options'] = vars(options['model_options'])
    if 'train_options' in options:
        options['train_options'] = vars(options['train_options'])
    if 'test_options' in options:
        options['test_options'] = vars(options['test_options'])

    with open(os.path.join(path, file_name), "w+") as f:
        f.write(json.dumps(options, indent=1))
