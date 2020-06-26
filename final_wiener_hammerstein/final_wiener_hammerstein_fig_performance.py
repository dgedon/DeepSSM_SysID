import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pandas as pd
import sys
# only to compute the performance if not available
import data.loader as loader
from models.model_state import ModelState
import testing
from utils.utils import compute_normalizer
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params

os.chdir('../')
sys.path.append(os.getcwd())


# %% get performance results if not available

def get_perf_results(path_general, model_name):
    options = {
        'dataset': 'wiener_hammerstein',
        'model': model_name,
        'logdir': 'final',
        'normalize': True,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': False,
        'savefig': False,
        'MCsamples': 20,
        'gridvalues': {
            'h_values': [30, 40, 50, 60, 70],
            'z_values': [3],
            'n_values': [3], },
        'train_set': 'small',
    }
    h_values = options['gridvalues']['h_values']
    z_values = options['gridvalues']['z_values']
    n_values = options['gridvalues']['n_values']

    # options
    # get the options
    options['device'] = torch.device('cpu')
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['test_options'] = train_params.get_test_options()

    # file name
    file_name_general = dataset

    # allocation
    vaf_all_multisine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])
    rmse_all_multisine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])
    likelihood_all_multisine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])

    vaf_all_sweptsine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])
    rmse_all_sweptsine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])
    likelihood_all_sweptsine = torch.zeros([options['MCsamples'], len(h_values), len(z_values), len(n_values)])

    for mcIter in range(options['MCsamples']):
        print('\n#####################')
        print('MC ITERATION: {}/{}'.format(mcIter + 1, options['MCsamples']))
        print('#####################\n')

        for i1, h_sel in enumerate(h_values):
            for i2, z_sel in enumerate(z_values):
                for i3, n_sel in enumerate(n_values):

                    # output current choice
                    print('\nCurrent run: h={}, z={}, n={}\n'.format(h_sel, z_sel, n_sel))

                    # get curren file names
                    file_name = file_name_general + '_h{}_z{}_n{}_MC{}'.format(h_sel, z_sel, n_sel, mcIter)

                    # set new values in options
                    options['model_options'].h_dim = h_sel
                    options['model_options'].z_dim = z_sel
                    options['model_options'].n_layers = n_sel

                    # Specifying datasets (only matters for testing
                    kwargs = {'test_set': 'multisine', 'MCiter': mcIter, 'train_set': options['train_set']}
                    loaders_multisine = loader.load_dataset(dataset=options["dataset"],
                                                            dataset_options=options["dataset_options"],
                                                            train_batch_size=options["train_options"].batch_size,
                                                            test_batch_size=options["test_options"].batch_size,
                                                            **kwargs)

                    kwargs = {'test_set': 'sweptsine', 'MCiter': mcIter}
                    loaders_sweptsine = loader.load_dataset(dataset=options["dataset"],
                                                            dataset_options=options["dataset_options"],
                                                            train_batch_size=options["train_options"].batch_size,
                                                            test_batch_size=options["test_options"].batch_size,
                                                            **kwargs)

                    # Compute normalizers
                    if options["normalize"]:
                        normalizer_input, normalizer_output = compute_normalizer(loaders_multisine['train'])
                    else:
                        normalizer_input = normalizer_output = None

                    # Define model
                    modelstate = ModelState(seed=options["seed"],
                                            nu=loaders_multisine["train"].nu, ny=loaders_multisine["train"].ny,
                                            model=options["model"],
                                            options=options,
                                            normalizer_input=normalizer_input,
                                            normalizer_output=normalizer_output)
                    modelstate.model.to(options['device'])

                    # allocation
                    df = {}

                    # test the model
                    print('\nTest: Multisine')
                    kwargs = {'file_name_add': 'Multisine_'}
                    df_multisine = df
                    df_multisine = testing.run_test(options, loaders_multisine, df_multisine, path_general, file_name,
                                                    **kwargs)
                    print('\nTest: Sweptsine')
                    kwargs = {'file_name_add': 'Sweptsine_'}
                    df_sweptsine = {}
                    df_sweptsine = testing.run_test(options, loaders_sweptsine, df_sweptsine, path_general, file_name,
                                                    **kwargs)

                    # save performance values
                    vaf_all_multisine[mcIter, i1, i2, i3] = df_multisine['vaf']
                    rmse_all_multisine[mcIter, i1, i2, i3] = df_multisine['rmse'][0]
                    likelihood_all_multisine[mcIter, i1, i2, i3] = df_multisine['marginal_likeli'].item()

                    vaf_all_sweptsine[mcIter, i1, i2, i3] = df_sweptsine['vaf']
                    rmse_all_sweptsine[mcIter, i1, i2, i3] = df_sweptsine['rmse'][0]
                    likelihood_all_sweptsine[mcIter, i1, i2, i3] = df_sweptsine['marginal_likeli'].item()
    # save data
    datasaver = {'all_vaf_multisine': vaf_all_multisine,
                 'all_rmse_multisine': rmse_all_multisine,
                 'all_likelihood_multisine': likelihood_all_multisine,
                 'all_vaf_sweptsine': vaf_all_sweptsine,
                 'all_rmse_sweptsine': rmse_all_sweptsine,
                 'all_likelihood_sweptsine': likelihood_all_sweptsine}
    # get saving path
    path = path_general + 'data/'
    # filename
    file_name = '{}.pt'.format(options['dataset'])
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # save data
    torch.save(datasaver, path + file_name)

    print('\n')
    print('# ' * 20)
    print('Performance computation for model {}: DONE'.format(model_name))
    print('# ' * 20)
    print('\n')


# %%  #######################################################################

if __name__ == "__main__":

    # set (high level) options dictionary
    dataset = 'wiener_hammerstein'
    logdir = 'final'
    addlog = 'run_9999_final'  # 'run_0317_hvar_new'
    model = ['VAE-RNN', 'VRNN-Gauss-I', 'VRNN-Gauss', 'VRNN-GMM-I', 'VRNN-GMM', 'STORN']
    varying_param = 'hvary'

    # %% allocation
    if varying_param == 'zvary':
        x_values = np.array([1, 3, 5, 7])
    elif varying_param == 'hvary':
        x_values = np.array([30, 40, 50, 60, 70])

    rmse_all_multisine = []
    rmse_all_sweptsine = []
    nll_all_multisine = []
    nll_all_sweptsine = []

    # %% get the likelihood and rmse matrices

    for i, model_sel in enumerate(model):
        # get saving path
        path_general = os.getcwd() + '/log_Server/{}/{}/{}/{}/'.format(logdir, dataset, addlog, model_sel)
        # get data
        file_name = dataset + '.pt'
        path = path_general + 'data/'
        try:
            data = torch.load(path + file_name)
        except:  # run all available models, collect the data and save it in '/data'
            get_perf_results(path_general, model_sel)
            data = torch.load(path + file_name)
        stopvar = 1

        # load the data: RMSE
        rmse_all_multisine.append(data['all_rmse_multisine'])
        rmse_all_sweptsine.append(data['all_rmse_sweptsine'])
        # load the data: NLL
        nll_all_multisine.append(-data['all_likelihood_multisine'])
        nll_all_sweptsine.append(-data['all_likelihood_sweptsine'])

    # %% plot everything

    plt.figure(figsize=(5 * 2, 5 * 1))
    # plot rmse: multisine
    plt.subplot(1, 2, 1)
    for i, model_sel in enumerate(model):
        mu = rmse_all_multisine[i].mean(0).squeeze()
        std = np.sqrt(rmse_all_multisine[i].var(0)).squeeze()
        # plot mean
        plt.plot(x_values, mu, label=model_sel)
        # plot std
        # plt.fill_between(ndata, mu, mu + std, alpha=0.3, facecolor='b')
        # plt.fill_between(ndata, mu, mu - std, alpha=0.3, facecolor='b')
    plt.legend()
    plt.xlabel('training data points')
    plt.ylabel('RMSE')
    plt.title('RMSE of multisine')

    # plot rmse: sweptsine
    plt.subplot(1, 2, 2)
    for i, model_sel in enumerate(model):
        mu = rmse_all_sweptsine[i].mean(0).squeeze()
        std = np.sqrt(rmse_all_sweptsine[i].var(0)).squeeze()
        # plot mean
        plt.plot(x_values, mu, label=model_sel)
        # plot std
        # plt.fill_between(ndata, mu, mu + std, alpha=0.3, facecolor='b')
        # plt.fill_between(ndata, mu, mu - std, alpha=0.3, facecolor='b')
    plt.legend()
    plt.xlabel('training data points')
    plt.ylabel('RMSE')
    plt.title('RMSE of sweptsine')

    plt.show()

    # %% output of best values

    for i, model_sel in enumerate(model):
        mean_rmse_multisine = rmse_all_multisine[i].mean(0).squeeze()
        mean_rmse_multisine_idx = np.argmin(mean_rmse_multisine)

        mean_rmse_sweptsine = rmse_all_sweptsine[i].mean(0).squeeze()
        mean_rmse_sweptsine_idx = np.argmin(mean_rmse_sweptsine)

        print(model_sel)
        print('\tmin Multisine: RMSE={} at x={}'.format(mean_rmse_multisine[mean_rmse_multisine_idx],
                                                        x_values[mean_rmse_multisine_idx]))
        print('\tmin Sweptsine RMSE={} at x={}\n'.format(mean_rmse_sweptsine[mean_rmse_sweptsine_idx],
                                                         x_values[mean_rmse_sweptsine_idx]))

    # %% save data for pgfplots

    data = {'x': x_values, }

    for i, model_sel in enumerate(model):
        # RMSE: multisine
        mean_rmse_multisine = rmse_all_multisine[i].mean(0).squeeze().numpy()
        std_rmse_multisine = np.sqrt(rmse_all_multisine[i].var(0)).squeeze().numpy()
        update_mu_rmse_multisine = {'mu_rmse_ms_{}'.format(model[i]): mean_rmse_multisine}
        update_pstd_rmse_multisine = {'pstd_rmse_ms_{}'.format(model[i]): mean_rmse_multisine + std_rmse_multisine}
        update_mstd_rmse_multisine = {'mstd_rmse_ms_{}'.format(model[i]): mean_rmse_multisine - std_rmse_multisine}

        # update dictionary
        data.update(update_mu_rmse_multisine)
        data.update(update_pstd_rmse_multisine)
        data.update(update_mstd_rmse_multisine)

        # RMSE: Swept sine
        mean_rmse_sweptsine = rmse_all_sweptsine[i].mean(0).squeeze().numpy()
        std_rmse_sweptsine = np.sqrt(rmse_all_sweptsine[i].var(0)).squeeze().numpy()
        update_mu_rmse_sweptsine = {'mu_rmse_ss_{}'.format(model[i]): mean_rmse_sweptsine}
        update_pstd_rmse_sweptsine = {'pstd_rmse_ss_{}'.format(model[i]): mean_rmse_sweptsine + std_rmse_sweptsine}
        update_mstd_rmse_sweptsine = {'mstd_rmse_ss_{}'.format(model[i]): mean_rmse_sweptsine - std_rmse_sweptsine}

        # update dictionary
        data.update(update_mu_rmse_sweptsine)
        data.update(update_pstd_rmse_sweptsine)
        data.update(update_mstd_rmse_sweptsine)

    df = pd.DataFrame(data)

    path = os.getcwd() + '/final_wiener_hammerstein/' + 'WH_data_performance_' + varying_param + '_' + addlog + '.csv'
    df.to_csv(path, index=False)
