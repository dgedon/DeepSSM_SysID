from pathlib import Path
import scipy.io
import numpy as np
import os
import urllib
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from data.base import IODataset


def create_f16gvt_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None,):
    """Load f16gvt data: train, validation and test datasets.
    Parameters
    ----------
    seq_len_train, seq_len_val, seq_len_test: int (optional)
        Maximum length for a batch on, respectively, the training,
        validation and test sets. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If None,
        put the entire dataset on a single batch.
    seq_len_eval: int (optional)
        Maximum length for a batch on the validatiteston and  set. If `seq_len`
        is smaller than the total data length, the training data will
        be further divided in batches
    Returns
    -------
    dataset_train, dataset_val, dataset_test: Dataset
        Train, validation and test data
    Note
    ----
    Based on https://github.com/locuslab/TCN/blob/master/TCN/lambada_language/utils.py
    """
    # Extract input and output data F16
    mat_file_path = "F16Data_SpecialOddMSine_Level2.mat"
    mat_two = scipy.io.loadmat(maybe_download_and_extract(mat_file_path))
    mat_file_path = "F16Data_SpecialOddMSine_Level2_Validation.mat"
    mat_two_test = scipy.io.loadmat(maybe_download_and_extract(mat_file_path))
    u = mat_two['Force']  # Input training and validation set
    y = mat_two['Acceleration']  # Output training and validation set
    u_test = mat_two_test['Force']  # Input test set
    y_test = mat_two_test['Acceleration']  # Output test set

    # Properties data
    n_pp = 16384  # Number of samples per period and per multisine realization
    ptrans = 1  # One transient period
    p = 2  # Last two periods are in steady-state
    n = (ptrans + p) * n_pp  # Number of samples per multisine realization
    r = 8  # Number of multisine realizations used for training
    r_val = 9 - r  # Number of multisine realizations used for validation
    r_test = 1  # One multisine realization in test set
    n_y = 3  # Number of outputs

    # Extract training data
    u_train = u[0:r, :].reshape(r * n, 1)
    y_train = y[:, 0:r, :].transpose((1, 2, 0)).reshape((r * n, n_y))

    # Extract validation data
    u_val = u[r + np.arange(r_val), :].reshape(r_val * n, 1)
    y_val = y[:, r + np.arange(r_val), :].transpose((1, 2, 0)).reshape((r_val * n, n_y))

    # Extract test data
    u_test = u_test.reshape(r_test * n, 1)
    y_test = y_test.transpose((2, 1, 0)).reshape((r_test * n, n_y))

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test


def maybe_download_and_extract(mat_file_path):
    """Download the data from nonlinear benchmark website, unless it's already here."""
    src_url = 'http://nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16GVT_Files.zip'
    home = Path.home()
    work_dir = 'data/F16GVT_Files'  # str(home.joinpath('datasets/F16Gvt'))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    zipfilepath = os.path.join(work_dir, "F16GVT_Files.zip")
    if not os.path.exists(zipfilepath):
        filepath, _ = urllib.request.urlretrieve(
            src_url, zipfilepath)
        file = os.stat(filepath)
        size = file.st_size
        print('Successfully downloaded', 'F16GVT_Files.zip', size, 'bytes.')
    else:
        print('F16GVT_Files.zip', 'already downloaded!')

    datafilepath = os.path.join(work_dir, mat_file_path)
    print(datafilepath)
    if not os.path.exists(datafilepath):
        zip_ref = zipfile.ZipFile(zipfilepath, 'r')
        zip_ref.extractall(work_dir)
        zip_ref.close()
        print('Successfully unzipped data')
    return datafilepath


if __name__ == "__main__":
    # For testing purposes, should be removed afterwards
    train, val, test = create_f16gvt_datasets(seq_len_train=1000)
    # Convert back from torch tensor to numpy vector
    u_train = train.u.reshape(-1)
    y_train = train.y.reshape((-1, 3))
    k_train = 1 + np.arange(len(u_train))
    u_val = val.u.reshape(-1)
    y_val = val.y.reshape((-1, 3))
    k_val = 1 + np.arange(len(u_val))
    u_test = test.u.reshape(-1)
    y_test = test.y.reshape((-1, 3))
    k_test = 1 + np.arange(len(u_test))
    # Plot training data
    plt.figure()
    plt.plot(k_train, u_train)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input training data')
    plt.show()
    plt.figure()
    plt.plot(k_train, y_train)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output training data')
    plt.show()
    # Plot validation data
    plt.figure()
    plt.plot(k_val, u_val)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input validation data')
    plt.show()
    plt.figure()
    plt.plot(k_val, y_val)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output validation data')
    plt.show()
    # Plot test data
    plt.figure()
    plt.plot(k_test, u_test)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input test data')
    plt.show()
    plt.figure()
    plt.plot(k_test, y_test)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output test data')
    plt.show()