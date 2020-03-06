# import generic libraries
import matplotlib.pyplot as plt
import torch.utils.data
import os
import numpy as np
import time
import sys

os.chdir('../')
sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
from models.model_state import ModelState
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
import utils.datavisualizer as dv

# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params


def print_best_values(all_rmse, all_vaf, all_likelihood, h_values, z_values, n_values):
    all_vaf = all_vaf.numpy()
    i, j, k, l = np.unravel_index(all_vaf.argmax(), all_vaf.shape)
    print('Best Parameters max vaf={}, h={}, z={}, n={}, MC={}, ind(h,z,n)=({},{},{})'.format(all_vaf[i, j, k, l],
                                                                                              h_values[j],
                                                                                              z_values[k],
                                                                                              n_values[l], i, j, k, l))
    all_rmse = all_rmse.numpy()
    i, j, k, l = np.unravel_index(all_rmse.argmin(), all_rmse.shape)
    print('Best Parameters min rmse={}, h={}, z={}, n={}, MC={}, ind(h,z,n)=({},{},{})'.format(all_rmse[i, j, k, l],
                                                                                               h_values[j],
                                                                                               z_values[k],
                                                                                               n_values[l], i, j, k, l))
    all_likelihood = all_likelihood.numpy()
    i, j, k, l = np.unravel_index(all_likelihood.argmax(), all_likelihood.shape)
    print('Best Parameters max likelihood={}, h={}, z={}, n={}, MC={}, ind(h,z,n)=({},{},{})'.format(
        all_likelihood[i, j, k, l],
        h_values[j],
        z_values[k],
        n_values[l], i, j, k, l))


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
    'MCsamples': 5,
    'gridvalues': {
        'h_values': [60],  # [10, 20, 30, 40, 50, 60, 70, 80],
        'z_values': [10],
        'n_values': [2], },
}

addlog = 'run_0306_64_2'
# get saving path
path_general = os.getcwd() + '/log/{}/{}/{}/{}/'.format(options['logdir'],
                                                        options['dataset'],
                                                        addlog,
                                                        options['model'], )
# get saving file names
file_name_general = options['dataset']

# %%
if __name__ == "__main__":
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general + '_runlog')

    print('Run file: final_wiener_hammerstein.py')
    print(time.strftime("%c"))

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

    h_values = options['gridvalues']['h_values']
    z_values = options['gridvalues']['z_values']
    n_values = options['gridvalues']['n_values']

    # print number of searches
    temp = len(h_values) * len(z_values) * len(n_values)
    print('Total number of evaluation points: {}'.format(temp))

    # save options
    save_options(options, path_general, 'options.txt')

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

        # set the correct device to run on
        options['device'] = device

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
                    kwargs = {'test_set': 'multisine', 'MCiter': mcIter}
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

                    if options['do_train']:
                        # train the model
                        df = training.run_train(modelstate=modelstate,
                                                loader_train=loaders_multisine['train'],
                                                loader_valid=loaders_multisine['valid'],
                                                options=options,
                                                dataframe=df,
                                                path_general=path_general,
                                                file_name_general=file_name)

                    if options['do_test']:
                        # test the model
                        print('\nTest: Multisine')
                        kwargs = {'file_name_add': 'Multisine_'}
                        df_multisine = testing.run_test(options, loaders_multisine, df, path_general, file_name,
                                                        **kwargs)
                        print('\nTest: Sweptsine')
                        kwargs = {'file_name_add': 'Sweptsine_'}
                        df_sweptsine = testing.run_test(options, loaders_sweptsine, df, path_general, file_name,
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

    # output best parameters
    print('\nBest values: Multisine')
    print_best_values(rmse_all_multisine, vaf_all_multisine, likelihood_all_multisine, h_values, z_values, n_values)
    print('\nBest values: Swept sine')
    print_best_values(rmse_all_sweptsine, vaf_all_sweptsine, likelihood_all_sweptsine, h_values, z_values, n_values)

    # plot performance
    dv.plot_perf_sizes(h_values, vaf_all_multisine, rmse_all_multisine, likelihood_all_multisine, options, path_general)
    dv.plot_perf_sizes(h_values, vaf_all_sweptsine, rmse_all_sweptsine, likelihood_all_sweptsine, options, path_general)

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {:2.0f}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))
