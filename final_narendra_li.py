import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import csv
import pandas as pd

# %%
# set (high level) options dictionary
dataset = 'narendra_li'
logdir = 'ndata'
addlog = 'run200223'
model = ['VAE-RNN', 'VRNN-Gauss', 'VRNN-Gauss-I', 'VRNN-GMM', 'VRNN-GMM-I', 'STORN']

ndata = np.array([2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000])
all_rmse = np.zeros([len(model), len(ndata)])
all_likelihood = np.zeros([len(model), len(ndata)])

# %% get the likelihood and rmse matrices

for i, model_sel in enumerate(model):
    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}/{}/'.format(logdir, dataset, addlog, model_sel)

    # load the data: RMSE
    file_name = 'all_rmse.pt'
    path = path_general + 'data/' + file_name
    temp_rmse = torch.load(path)
    all_rmse[i, :] = temp_rmse.numpy()

    # load the data: VAF
    file_name = 'all_likelihood.pt'
    path = path_general + 'data/' + file_name
    temp_likelihood = torch.load(path)
    all_likelihood[i, :] = temp_likelihood.numpy()

# %% plot everything

plt.figure(figsize=(5 * 2, 5 * 1))
# plot rmse
plt.subplot(1, 2, 1)
for i, model_sel in enumerate(model):
    plt.plot(ndata, all_rmse[i, :], label=model_sel)
plt.legend()
plt.xlabel('training data points')
plt.ylabel('RMSE')
plt.title('RMSE of Narendra-Li Benchmark')

# plot rmse
plt.subplot(1, 2, 2)
for i, model_sel in enumerate(model):
    plt.plot(ndata, all_likelihood[i, :], label=model_sel)
plt.legend()
plt.xlabel('training data points')
plt.ylabel('log-likelihood')
plt.title('Log-Likelihood of Narendra-Li Benchmark')

plt.show()
