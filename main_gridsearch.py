# import generic libraries
import torch
import torch.utils.data
import pandas as pd
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


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_gridsearch(options, kwargs, gridvalues, path_general, file_name_general):
    print('Run file: main_gridsearch.py')
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
    set_redirects(path, file_name_general+'_runlog')

    h_values = gridvalues['h_values']
    z_values = gridvalues['z_values']
    n_values = gridvalues['n_values']

    # print number of searches
    temp = len(h_values) * len(z_values) * len(n_values)
    print('Total number of search points: {}'.format(temp))

    # allocation
    all_vaf = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_rmse = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_likelihood = torch.zeros([len(h_values), len(z_values), len(n_values)])
    all_df = {}

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

                """# get the non-trained model
                model = models.model_state.get_model(options)
                model = model.to(options['device'])"""

                # Specifying datasets
                loaders = loader.load_dataset(dataset=options["dataset"],
                                              dataset_options=options["dataset_options"],
                                              train_batch_size=options["train_options"].batch_size,
                                              test_batch_size=options["test_options"].batch_size,
                                              **kwargs)

                # Compute normalizers
                if options["normalize"]:
                    normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
                else:
                    normalizer_input = normalizer_output = None

                # Define model
                modelstate = ModelState(seed=options["seed"],
                                        nu=loaders["train"].nu, ny=loaders["train"].ny,
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
                                         loader_train=loaders['train'],
                                         loader_valid=loaders['valid'],
                                         options=options,
                                         dataframe=df,
                                         path_general=path_general,
                                         file_name_general=file_name)

                if options['do_test']:
                    # test the model
                    df = test.run_test(options, loaders, df, path_general, file_name)

                # store values
                all_df[(i1, i2, i3)] = df

                # save performance values
                all_vaf[i1, i2, i3] = df['vaf']
                all_rmse[i1, i2, i3] = df['rmse'][0]
                all_likelihood[i1, i2, i3] = df['marginal_likeli'].item()

    # save data
    # get saving path
    path = path_general + 'data/'
    # to pandas
    all_df = pd.DataFrame(all_df)
    # filename
    file_name = '{}_gridsearch.csv'.format(options['dataset'])
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)

    # save data
    all_df.to_csv(path_general + file_name)
    # save performance values
    torch.save(all_vaf, path_general + 'data/' + 'all_vaf.pt')
    torch.save(all_rmse, path_general + 'data/' + 'all_rmse.pt')
    torch.save(all_likelihood, path_general + 'data/' + 'all_likelihood.pt')

    # output best parameters
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

    # plot results
    dv.plot_perf_gridsearch(all_vaf, all_rmse, all_likelihood, z_values, h_values, path_general, options)

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {:2.0f}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))


# %%
if __name__ == "__main__":
    # set (high level) options dictionary
    options = {
        'dataset': 'narendra_li',  # 'f16gvt', 'cascaded_tank', 'narendra_li'
        'model': 'VRNN-Gauss',
        'do_train': True,
        'do_test': True,
        'logdir': 'gridsearch',
        'normalize': True,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': True,
        'savefig': True,
    }

    # select parameters for narendra-li benchmark
    kwargs = {"k_max_train": 50000,
              "k_max_val": 5000,
              "k_max_test": 5000}

    # values for grid search
    gridvalues = {
        'h_values': [10, 20, 30, 40, 50, 60, 70, 80],  # [40, 50]  # [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        'z_values': [1, 2, 5, 10],  # [2]  # [1, 2, 5, 10, 15, 20]
        'n_values': [1],  # [1]  # [1, 2, 5, 10]
    }

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'], )

    # get saving file names
    file_name_general = options['dataset']

    run_main_gridsearch(options, kwargs, gridvalues, path_general, file_name_general)
