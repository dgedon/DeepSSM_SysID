import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
from data.base import IODataset


# watertank benchmark
def get_cascaded_tank_sim_data(dynsys_param, k_max, mode):
    # parse input parameter
    x0 = dynsys_param.x0  # initial state
    y_dim = dynsys_param.y_dim  # output dimension
    u_dim = dynsys_param.u_dim  # input dimension
    sigma_w = dynsys_param.sigma_state  # noise variance of state
    sigma_v = dynsys_param.sigma_output  # noise variance of output
    param = dynsys_param.nonlin_param  # parameters of nonlinear system

    # parameters of nonlinear system
    k1 = param[0]
    k2 = param[1]
    k3 = param[2]
    k4 = param[3]
    x_max = [param[4], param[5]]  # cut-off of state

    # allocation
    x = np.zeros([2, k_max + 1])
    u = np.zeros([u_dim, k_max])
    y = np.zeros([y_dim, k_max])

    # initial values
    x[:, 0] = x0

    # simulate
    for k in range(0, k_max):
        # input signal
        if mode == 'train':
            # input u for training
            if k % 30 == 0:     # k % 7 == 0:
                u[0, k] = 5.5 * np.random.rand(1) - 1  # 5 * np.random.randn(1)
            else:
                u[0, k] = u[0, k - 1]
        elif mode == 'test':
            # input u for testing
            u[0, k] = 2.5*(2.5 * np.sin(k / 7) + 1.5 * np.cos(k / 3) + 1)   # (2.5 * np.sin(k / 7) + 1.5 * np.cos(k / 3) + 1)

        # state evolution
        # true dynamics from benchmark paper
        temp = x[:, k] + np.array([-k1 * np.sqrt(x[0, k]) + k2 * u[0, k],
                                   k3 * np.sqrt(x[0, k]) - k4 * np.sqrt(x[1, k])]) + sigma_w * np.random.randn(2)

        # include additional non-linearity by capping at zero and top of water tank
        x[:, k + 1] = np.minimum(x_max, np.maximum(np.zeros([2]), temp))

        # output measurement
        y[:, k] = x[1, k] + sigma_v * np.random.randn(y_dim)

    # return values
    return torch.tensor(y), torch.tensor(u)  # torch.tensor(x),


def get_cascaded_tank_data():
    # data file direction and name
    file_name = 'data/CascadedTanksFiles/dataBenchmark.csv'

    # initialization
    u_train = []
    u_test = []
    y_train = []
    y_test = []

    # read the file into variable
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # ignore header line
            if line_count == 0:
                line_count += 1
            else:
                #
                u_train.append(float(row[0]))
                u_test.append(float(row[1]))
                y_train.append(float(row[2]))
                y_test.append(float(row[3]))

    # convert to torch tensors
    u_train = torch.FloatTensor([u_train])
    u_test = torch.FloatTensor([u_test])
    y_train = torch.FloatTensor([y_train])
    y_test = torch.FloatTensor([y_test])

    """plt.plot(y_train)
    plt.plot(y_test)
    plt.show()"""

    return y_train, u_train, y_test, u_test


def create_cascadedtank_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None,):
    # data file direction and name
    file_name = 'data/CascadedTanksFiles/dataBenchmark.csv'

    # initialization
    u = []
    y = []
    u_test = []
    y_test = []

    # read the file into variable
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # ignore header line
            if line_count == 0:
                line_count += 1
            else:
                # Extract combination of training / validation data
                u.append(float(row[0]))
                y.append(float(row[2]))
                # Extract test data
                u_test.append(float(row[1]))
                y_test.append(float(row[3]))

    # convert from list to numpy array
    u = np.asarray(u)
    y = np.asarray(y)
    u_test = np.asarray(u_test)
    y_test = np.asarray(y_test)

    # add some measurement noise to the data
    # sigma = 0  #.15
    y = y  # + sigma * np.random.randn(y.shape[0])
    y_test = y_test  # + sigma * np.random.randn(y_test.shape[0])

    # number of test data points
    p = 896  # 1024 - 4 * 32 (7 batches of training and 1 batch of validation)

    # Extract training data
    u_train = u[0:p]
    y_train = y[0:p]

    # Extract validation data
    u_val = u[p:]
    y_val = y[p:]

    # get correct dimensions
    u_test = u_test[..., None]
    y_test = y_test[..., None]
    u_train = u_train[..., None]
    y_train = y_train[..., None]
    u_val = u_val[..., None]
    y_val = y_val[..., None]

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test
