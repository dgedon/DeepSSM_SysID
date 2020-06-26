import matplotlib.pyplot as plt
import numpy as np
from data.base import IODataset


def run_toy_lgssm_sim(u, A, B, C, sigma_state, sigma_out):
    # just a standard linear gaussian state space model. Measurement Noise is considered outside
    # same system as in toy examples of "Learning of state-space models with highly informative observations: a tempered
    # Sequential Monte Carlo solution", chapter 5.1
    # additionally measurement noise considered

    # get length of input
    k_max = u.shape[-1]

    # size of variables
    n_u = 1
    n_y = 1
    n_x = 2

    # allocation
    x = np.zeros([n_x, k_max + 1])
    y = np.zeros([n_y, k_max])

    # run over all time steps
    for k in range(k_max):
        x[:, k + 1] = np.dot(A, x[:, k]) + np.dot(B, u[:, k]) + sigma_state * np.random.randn(n_x)
        y[:, k] = np.dot(C, x[:, k])

    return y


def create_toy_lgssm_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    # state space matrices
    A = np.array([[0.7, 0.8], [0, 0.1]])
    B = np.array([[-1], [0.1]])
    C = np.array([[1], [0]]).transpose()
    # define noise
    sigma_state = np.sqrt(0.25)
    sigma_out = np.sqrt(1)

    # length of all data sets
    if bool(kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 5000
        k_max_val = 5000
        k_max_test = 5000

    # training / validation set input
    u_train = (np.random.rand(1, k_max_train) - 0.5) * 5
    u_val = (np.random.rand(1, k_max_val) - 0.5) * 5
    # test set input
    file_path = 'data/Toy_LGSSM/toy_lgssm_testdata.npz'
    test_data = np.load(file_path)
    u_test = test_data['u_test'][0:k_max_test]
    y_test = test_data['y_test'][0:k_max_test]

    # get the outputs
    y_train = run_toy_lgssm_sim(u_train, A, B, C, sigma_state, 0) + sigma_out * np.random.randn(1, k_max_train)
    y_val = run_toy_lgssm_sim(u_val, A, B, C, sigma_state, 0) + sigma_out * np.random.randn(1, k_max_val)

    # get correct dimensions
    u_train = u_train.transpose(1, 0)
    y_train = y_train.transpose(1, 0)
    u_val = u_val.transpose(1, 0)
    y_val = y_val.transpose(1, 0)

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test
