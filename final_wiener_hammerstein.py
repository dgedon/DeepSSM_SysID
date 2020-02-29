# import generic libraries
import matplotlib.pyplot as plt
import torch.utils.data
import os
import numpy as np
import time
import utils.datavisualizer as dv
# import user-written files
import data.loader as loader
from models.model_state import ModelState
import train
import test
from utils.utils import compute_normalizer
from utils.logger import set_redirects

# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params


def print_best_values(all_rmse, all_vaf, all_likelihood, h_values, z_values, n_values):
    all_vaf = all_vaf.numpy()
    i, j, k = np.unravel_index(all_vaf.argmax(), all_vaf.shape)
    print('Best Parameters max vaf={}, h={}, z={}, n={}, ind(h,z,n)=({},{},{})'.format(all_vaf[i, j, k],
                                                                                       h_values[i],
                                                                                       z_values[j],
                                                                                       n_values[k], i, j, k))
    all_rmse = all_rmse.numpy()
    i, j, k = np.unravel_index(all_rmse.argmin(), all_rmse.shape)
    print('Best Parameters min rmse={}, h={}, z={}, n={}, ind(h,z,n)=({},{},{})'.format(all_rmse[i, j, k],
                                                                                        h_values[i],
                                                                                        z_values[j],
                                                                                        n_values[k], i, j, k))
    all_likelihood = all_likelihood.numpy()
    i, j, k = np.unravel_index(all_likelihood.argmax(), all_likelihood.shape)
    print('Best Parameters max likelihood={}, h={}, z={}, n={}, ind(h,z,n)=({},{},{})'.format(all_likelihood[i, j, k],
                                                                                              h_values[i],
                                                                                              z_values[j],
                                                                                              n_values[k], i, j, k))


# set (high level) options dictionary
options = {
    'dataset': 'wiener_hammerstein',
    'model': 'STORN',
    'do_train': True,
    'do_test': True,
    'logdir': 'final',
    'normalize': True,
    'seed': 1234,
    'optim': 'Adam',
    'showfig': False,
    'savefig': True,
}

# values for grid search
gridvalues = {
    'h_values': [30, 40, 50, 60],  # [10, 20, 30, 40, 50, 60, 70, 80],
    'z_values': [15],
    'n_values': [1],
}

# get saving path
path_general = os.getcwd() + '/log/{}/{}/{}/'.format(options['logdir'],
                                                     options['dataset'],
                                                     options['model'], )
# get saving file names
file_name_general = options['dataset']

# %%
if __name__ == "__main__":

    print('Run file: final_wiener_hammerstein.py')
    start_time = time.time()
    # get correct computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('Device: {}'.format(device))

    # get the options
    options['device'] = device
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['test_options'] = train_params.get_test_options()

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general + '_runlog')

    h_values = gridvalues['h_values']
    z_values = gridvalues['z_values']
    n_values = gridvalues['n_values']

    # print number of searches
    temp = len(h_values) * len(z_values) * len(n_values)
    print('Total number of search points: {}'.format(temp))

    # allocation
    all_vaf_multisine = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_rmse_multisine = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_likelihood_multisine = torch.zeros([len(h_values), len(z_values), len(n_values)])

    all_vaf_sweptsine = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_rmse_sweptsine = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_likelihood_sweptsine = torch.zeros([len(h_values), len(z_values), len(n_values)])

    # MISSING: Loop over all models
    for i1, h_sel in enumerate(h_values):
        for i2, z_sel in enumerate(z_values):
            for i3, n_sel in enumerate(n_values):

                # output current choice
                print('\nCurrent run: h={}, z={}, n={}\n'.format(h_sel, z_sel, n_sel))

                # get curren file names
                file_name = file_name_general + '_h{}_z{}_n{}'.format(h_sel, z_sel, n_sel)

                # set new values in options
                options['model_options'].h_dim = h_sel
                options['model_options'].z_dim = z_sel
                options['model_options'].n_layers = n_sel

                # Specifying datasets (only matters for testing
                kwargs = {'test_set': 'multisine'}
                loaders_multisine = loader.load_dataset(dataset=options["dataset"],
                                                        dataset_options=options["dataset_options"],
                                                        train_batch_size=options["train_options"].batch_size,
                                                        test_batch_size=options["test_options"].batch_size,
                                                        **kwargs)

                kwargs = {'test_set': 'sweptsine'}
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

                if options['do_train']:
                    # train the model
                    df = train.run_train(modelstate=modelstate,
                                         loader_train=loaders_multisine['train'],
                                         loader_valid=loaders_multisine['valid'],
                                         options=options,
                                         dataframe=df,
                                         path_general=path_general,
                                         file_name_general=file_name)

                if options['do_test']:
                    # test the model
                    kwargs = {'file_name_add': 'Multisine_'}
                    df_multisine = test.run_test(options, loaders_multisine, {}, path_general, file_name, **kwargs)
                    kwargs = {'file_name_add': 'Sweptsine_'}
                    df_sweptsine = test.run_test(options, loaders_sweptsine, {}, path_general, file_name, **kwargs)

                # save performance values
                all_vaf_multisine[i1, i2, i3] = df_multisine['vaf']
                all_rmse_multisine[i1, i2, i3] = df_multisine['rmse'][0]
                all_likelihood_multisine[i1, i2, i3] = df_multisine['marginal_likeli'].item()

                all_vaf_sweptsine[i1, i2, i3] = df_sweptsine['vaf']
                all_rmse_sweptsine[i1, i2, i3] = df_sweptsine['rmse'][0]
                all_likelihood_sweptsine[i1, i2, i3] = df_sweptsine['marginal_likeli'].item()

    # save data
    datasaver = {'all_vaf_multisine': all_vaf_multisine,
                 'all_rmse_multisine': all_rmse_multisine,
                 'all_likelihood_multisine': all_likelihood_multisine,
                 'all_vaf_sweptsine': all_vaf_sweptsine,
                 'all_rmse_sweptsine': all_rmse_sweptsine,
                 'all_likelihood_sweptsine': all_likelihood_sweptsine}
    # get saving path
    path = path_general + 'data/'
    # filename
    file_name = '{}_final.pt'.format(options['dataset'])
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # save data
    torch.save(datasaver, path + file_name)

    # output best parameters
    print('\nBest values: Multisine')
    print_best_values(all_rmse_multisine, all_vaf_multisine, all_likelihood_multisine, h_values, z_values, n_values)
    print('\nBest values: Swept sine')
    print_best_values(all_rmse_sweptsine, all_vaf_sweptsine, all_likelihood_sweptsine, h_values, z_values, n_values)

    # plot results
    plt.figure(figsize=(5 * 2, 5 * 1))
    # plot for multisine
    plt.subplot(1, 2, 1)
    plt.plot(np.asarray(h_values), all_rmse_multisine.numpy().squeeze())
    plt.xlabel('dimension of h')
    plt.ylabel('RMSE')
    plt.title('Multisine RMSE')
    # plot for swept sine
    plt.subplot(1, 2, 2)
    plt.plot(np.asarray(h_values), all_rmse_sweptsine.numpy().squeeze())
    plt.xlabel('dimension of h')
    plt.ylabel('RMSE')
    plt.title('Swept Sine RMSE')
    # saving path and file name
    path = path_general + 'Performance/'
    file_name = 'performanceEval'
    # save figure
    if options['savefig']:
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {:2.0f}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
