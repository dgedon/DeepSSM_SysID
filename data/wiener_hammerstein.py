import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
from data.base import IODataset


def create_wienerhammerstein_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    # which data set to use
    if 'test_set' in kwargs:
        test_set = kwargs['test_set']
    else:
        test_set = 'multisine'

    if 'train_set' in kwargs:
        train_set = kwargs['train_set']
    else:
        train_set = 'small'

    if 'MCiter' in kwargs:
        MCiter = kwargs['MCiter']
    else:
        MCiter = 0

    if test_set == 'multisine':
        test_idx = [2, 4]
    elif test_set == 'sweptsine':
        test_idx = [3, 5]

    # data file direction and name
    if train_set == 'small':
        file_name_train = 'data/WienerHammersteinFiles/WH_MultisineFadeOut.csv'
    elif train_set == 'big':
        file_name_train = 'data/WienerHammersteinFiles/WH_SineSweepInput_meas.csv'
    file_name_test = 'data/WienerHammersteinFiles/WH_TestDataset.csv'

    # initialization
    u = []
    y = []
    u_val = []
    y_val = []
    u_test = []
    y_test = []

    # read the file into variable
    with open(file_name_train, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # ignore header line
            if line_count == 0:
                line_count += 1
            else:
                # Extract combination of training / validation data
                if file_name_train == 'data/WienerHammersteinFiles/WH_SineSweepInput_meas.csv':
                    idx = 100 + MCiter
                    u.append(float(row[idx]))
                    y.append(float(row[2 * idx]))
                    u_val.append(float(row[idx + 1]))
                    y_val.append(float(row[2 * idx + 1]))
                elif file_name_train == 'data/WienerHammersteinFiles/WH_MultisineFadeOut.csv':
                    idx = 2
                    if MCiter % 2:
                        idx_add = 0
                    else:
                        idx_add = 1
                    u.append(float(row[idx + idx_add]))
                    y.append(float(row[2 * idx + idx_add]))
                    u_val.append(float(row[idx + 1 - idx_add]))
                    y_val.append(float(row[2 * idx + 1 - idx_add]))

    # read the file into variable
    with open(file_name_test, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # ignore header line
            if line_count == 0:
                line_count += 1
            else:
                # Extract combination of training / validation data
                u_test.append(float(row[test_idx[0]]))  # use 2,4 for multisine
                y_test.append(float(row[test_idx[1]]))  # use 3,5 for swept sine

    # convert from list to numpy array
    u_train = np.asarray(u)
    y_train = np.asarray(y)
    u_val = np.asarray(u_val)
    y_val = np.asarray(y_val)
    u_test = np.asarray(u_test)
    y_test = np.asarray(y_test)

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
