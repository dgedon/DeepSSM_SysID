import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class DataLoaderExt(DataLoader):
    @property
    def nu(self):
        return self.dataset.nu

    @property
    def ny(self):
        return self.dataset.ny


class IODataset(Dataset):
    """Create dataset from data.
    Parameters
    ----------
    u, y: ndarray, shape (total_len, n_channels) or (total_len,)
        Input and output signals. It should be either a 1d array or a 2d array.
    seq_len: int (optional)
        Maximum length for a batch on, respectively. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If None,
        put the entire dataset on a single batch.
    """
    def __init__(self, u, y, seq_len=None):
        if seq_len is None:
            seq_len = u.shape[0]
        self.u = IODataset._batchify(u.astype(np.float32), seq_len)
        self.y = IODataset._batchify(y.astype(np.float32), seq_len)
        self.ntotbatch = self.u.shape[0]
        self.seq_len = self.u.shape[2]
        self.nu = 1 if u.ndim == 1 else u.shape[1]
        self.ny = 1 if y.ndim == 1 else y.shape[1]

    def __len__(self):
        return self.ntotbatch

    def __getitem__(self, idx):
        return self.u[idx, ...], self.y[idx, ...]

    @staticmethod
    def _batchify(x, seq_len):
        # data should be a torch tensor
        # data should have size (total number of samples) times (number of signals)
        # The output has size (number of batches) times (number of signals) times (batch size)
        # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
        nbatch = x.shape[0] // seq_len
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        x = x[:nbatch * seq_len]
        #    data = data[range(nbatch * batch_size), :]
        #    data = np.reshape(data, [batch_size, nbatch, -1], order='F')
        #    data = np.transpose(data, (1, 2, 0))
        # Evenly divide the data across the batch_size batches and make sure it is still in temporal order
        #    data = data.reshape((nbatch, 1, seq_len)).transpose(0, 1, 2)
        x = x.reshape((seq_len, nbatch, -1), order='F').transpose(1, 2, 0)
        # data = data.view(nbatch, batch_size, -1).transpose(0, 1)
        # ## arg = np.zeros([1, 2], dtype=np.float32)

        return x
