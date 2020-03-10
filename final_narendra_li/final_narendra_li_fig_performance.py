import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import csv
import pandas as pd
import sys
os.chdir('../')
sys.path.append(os.getcwd())

# %%
# set (high level) options dictionary
dataset = 'narendra_li'
logdir = 'final'
addlog = 'run_0306_full'
model = ['VAE-RNN', 'VRNN-Gauss-I', 'VRNN-Gauss', 'VRNN-GMM-I', 'VRNN-GMM', 'STORN']

ndata = np.array([2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000])
rmse_all = []
nll_all = []

# %% get the likelihood and rmse matrices

for i, model_sel in enumerate(model):
    # get saving path
    path_general = os.getcwd() + '/log_Server/{}/{}/{}/{}/'.format(logdir, dataset, addlog, model_sel)
    # get data
    file_name = dataset + '.pt'
    path = path_general + 'data/'
    data = torch.load(path + file_name)

    # load the data: RMSE
    rmse_all.append(data['rmse_all'])
    # load the data: NLL
    nll_all.append(-data['likelihood_all'])

# %% plot everything

plt.figure(figsize=(5 * 2, 5 * 1))
# plot rmse
plt.subplot(1, 2, 1)
for i, model_sel in enumerate(model):
    mean = rmse_all[i].mean(0).squeeze()
    std = np.sqrt(rmse_all[i].var(0)).squeeze()
    # plot mean
    plt.plot(ndata, mean, label=model_sel)
    # plot std
    #plt.fill_between(ndata, mean, mean + std, alpha=0.3, facecolor='b')
    #plt.fill_between(ndata, mean, mean - std, alpha=0.3, facecolor='b')
plt.legend()
plt.xlabel('training data points')
plt.ylabel('RMSE')
plt.title('RMSE of Narendra-Li Benchmark')

# plot rmse
plt.subplot(1, 2, 2)
for i, model_sel in enumerate(model):
    mean = nll_all[i].mean(0).squeeze()
    std = np.sqrt(nll_all[i].var(0)).squeeze()
    # plot mean
    plt.plot(ndata, mean, label=model_sel)
    # plot std
    #plt.fill_between(ndata, mean, mean + std, alpha=0.3, facecolor='b')
    #plt.fill_between(ndata, mean, mean - std, alpha=0.3, facecolor='b')
plt.legend()
plt.xlabel('training data points')
plt.ylabel('NLL')
plt.title('NLL of Narendra-Li Benchmark')

plt.show()

# %% output of best values

for i, model_sel in enumerate(model):
    mean_rmse = rmse_all[i].mean(0).squeeze()
    mean_rmse_idx = np.argmin(mean_rmse)

    mean_nll = nll_all[i].mean(0).squeeze()
    mean_nll_idx = np.argmin(mean_nll)

    print(model_sel)
    print('\tmin RMSE={} at nData={}'.format(mean_rmse[mean_rmse_idx],ndata[mean_rmse_idx]))
    print('\tmin NLL={} at nData={}\n'.format(mean_nll[mean_nll_idx],ndata[mean_nll_idx]))

# %% save data for pgfplots

data = {'x': ndata, }

for i, model_sel in enumerate(model):
    # RMSE
    mean_rmse = rmse_all[i].mean(0).squeeze().numpy()
    std_rmse = np.sqrt(rmse_all[i].var(0)).squeeze().numpy()
    update_mu_rmse = {'mu_rmse_{}'.format(model[i]): mean_rmse}
    update_pstd_rmse = {'pstd_rmse_{}'.format(model[i]): mean_rmse+std_rmse}
    update_mstd_rmse = {'mstd_rmse_{}'.format(model[i]): mean_rmse-std_rmse}

    # update dictionary
    data.update(update_mu_rmse)
    data.update(update_pstd_rmse)
    data.update(update_mstd_rmse)

    # NLL
    mean_nll = nll_all[i].mean(0).squeeze().numpy()
    std_nll = np.sqrt(nll_all[i].var(0)).squeeze().numpy()
    update_mu_nll = {'mu_nll_{}'.format(model[i]): mean_nll}
    update_pstd_nll = {'pstd_nll_{}'.format(model[i]): mean_nll + std_nll}
    update_mstd_nll = {'mstd_nll_{}'.format(model[i]): mean_nll - std_nll}

    # update dictionary
    data.update(update_mu_nll)
    data.update(update_pstd_nll)
    data.update(update_mstd_nll)

df = pd.DataFrame(data)

path = os.getcwd() + '/final_narendra_li/' + 'narendra_li_data_performance.csv'
df.to_csv(path, index=False)
