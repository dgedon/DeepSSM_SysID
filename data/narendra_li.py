import matplotlib.pyplot as plt
import torch
import numpy as np
from data.base import IODataset


def run_narendra_li_sim(u):
    # see andreas lindholm's work "A flexible state-space model for learning nonlinear dynamical systems"

    # get length of input
    k_max = u.shape[-1]

    # allocation
    x = np.zeros([2, k_max + 1])
    y = np.zeros([1, k_max])

    # run over all time steps
    for k in range(k_max):
        # state 1
        x[0, k + 1] = (x[0, k] / (1 + x[0, k] ** 2) + 1) * np.sin(x[1, k])
        # state 2
        term1 = x[1, k] * np.cos(x[1, k])
        term2 = x[0, k] * np.exp(-1 / 8 * (x[0, k] ** 2 + x[1, k] ** 2))
        term3 = u[0, k] ** 3 / (1 + u[0, k] ** 2 + 0.5 * np.cos(x[0, k] + x[1, k]))
        x[1, k + 1] = term1 + term2 + term3
        # output
        term1 = x[0, k] / (1 + 0.5 * np.sin(x[1, k]))
        term2 = x[1, k] / (1 + 0.5 * np.sin(x[0, k]))
        y[0, k] = term1 + term2

    return y


def create_narendra_li_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    # define output noise
    sigma_out = np.sqrt(0.1)

    # length of all data sets
    if bool(kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 50000
        k_max_val = 5000
        k_max_test = 5000

    # training / validation set input
    u_train = (np.random.rand(1, k_max_train) - 0.5) * 5
    u_val = (np.random.rand(1, k_max_val) - 0.5) * 5
    # test set input
    file_path = 'data/Narendra_Li/narendra_li_testdata.npz'
    test_data = np.load(file_path)
    u_test = test_data['u_test'][0:k_max_test]
    y_test = test_data['y_test'][0:k_max_test]

    # get the outputs
    y_train = run_narendra_li_sim(u_train) + sigma_out * np.random.randn(1, k_max_train)
    y_val = run_narendra_li_sim(u_val) + sigma_out * np.random.randn(1, k_max_val)

    # get correct dimensions

    u_train = u_train.transpose(1, 0)
    y_train = y_train.transpose(1, 0)
    u_val = u_val.transpose(1, 0)
    y_val = y_val.transpose(1, 0)

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test
