import torch
import torch.nn as nn
from enum import Enum
import numpy as np


class Normalizer1D(nn.Module):
    _epsilon = 1e-16

    def __init__(self, scale, offset):
        super(Normalizer1D, self).__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32) + self._epsilon)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def normalize(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.offset) / self.scale
        return x.permute(0, 2, 1)

    def unnormalize(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.scale + self.offset
        return x.permute(0, 2, 1)

    def unnormalize_mean(self, x_mu):
        x_mu = x_mu.permute(0, 2, 1)
        x_mu = x_mu * self.scale + self.offset
        return x_mu.permute(0, 2, 1)

    def unnormalize_sigma(self, x_sigma):
        x_sigma = x_sigma.permute(0, 2, 1)
        x_sigma = x_sigma * self.scale
        return x_sigma.permute(0, 2, 1)


class DynamicModule(nn.Module):
    def __init__(self):
        super(DynamicModule, self).__init__()
        self.has_internal_state = False

    def get_requested_input(self, requested_output='internal'):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def init_hidden(self, batch_size, device=None):
        if self.has_internal_state:
            raise NotImplementedError

