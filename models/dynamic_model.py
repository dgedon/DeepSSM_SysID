import torch
import torch.nn as nn
import torch.nn.functional as F
from . import STORN, VAE_RNN, VRNN_Gauss, VRNN_Gauss_I, VRNN_GMM, VRNN_GMM_I


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, options, normalizer_input=None, normalizer_output=None,
                 *args, **kwargs):
        super(DynamicModel, self).__init__()
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.args = args
        self.kwargs = kwargs
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output
        self.zero_initial_state = False

        model_options = options['model_options']

        # initialize the model
        if model == 'VRNN-Gauss':
            self.m = VRNN_Gauss(model_options, options['device'])
        elif model == 'VRNN-Gauss-I':
            self.m = VRNN_Gauss_I(model_options, options['device'])
        elif model == 'VRNN-GMM':
            self.m = VRNN_GMM(model_options, options['device'])
        elif model == 'VRNN-GMM-I':
            self.m = VRNN_GMM_I(model_options, options['device'])
        elif model == 'STORN':
            self.m = STORN(model_options, options['device'])
        elif model == 'VAE-RNN':
            self.m = VAE_RNN(model_options, options['device'])
        else:
            raise Exception("Unimplemented model")

    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def forward(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)

        loss = self.m(u, y)

        return loss

    def generate(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)

        y_sample, y_sample_mu, y_sample_sigma = self.m.generate(u)

        if self.normalizer_output is not None:
            y_sample = self.normalizer_output.unnormalize(y_sample)
        if self.normalizer_output is not None:
            y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
        if self.normalizer_output is not None:
            y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)

        return y_sample, y_sample_mu, y_sample_sigma
