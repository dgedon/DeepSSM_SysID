# import generic libraries
import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
import time
# import user-written files
import utils.datavisualizer as dv
import data.loader as loader
import models.model_state
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects

# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_ndata(options, vary_data, path_general, file_name_general, params):
    print('Run file: main_ndata.py')
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

    # set new values in options
    options['model_options'].h_dim = params['h_best']
    options['model_options'].z_dim = params['z_best']
    options['model_options'].n_layers = params['n_best']

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general + '_runlog')

    # values of evaluation
    k_max_train_values = vary_data['k_max_train_values']
    k_max_val_values = vary_data['k_max_val_values']
    k_max_test_values = vary_data['k_max_test_values']

    # print number of evaluations
    print('Total number of data point sets: {}'.format(len(k_max_train_values)))

    # allocation
    all_vaf = torch.zeros([len(k_max_train_values)])
    all_rmse = torch.zeros([len(k_max_train_values)])
    all_likelihood = torch.zeros([len(k_max_train_values)])
    all_df = {}

    # MISSING: Loop over all models!!!
    for i, _ in enumerate(k_max_train_values):

        # output current choice
        print('\nCurrent run: k_max_train={}\n'.format(k_max_train_values[i]))

        # get current file name
        file_name = file_name_general + '_kmaxtrain_{}'.format(k_max_train_values[i])

        # select parameters
        kwargs = {"k_max_train": k_max_train_values[i],
                  "k_max_val": k_max_val_values[i],
                  "k_max_test": k_max_test_values[i]}

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
            df = training.run_train(modelstate=modelstate,
                                    loader_train=loaders['train'],
                                    loader_valid=loaders['valid'],
                                    options=options,
                                    dataframe=df,
                                    path_general=path_general,
                                    file_name_general=file_name, )

        if options['do_test']:
            # test the model
            df = testing.run_test(options, loaders, df, path_general, file_name)

        # store values
        all_df[i] = df

        # save performance values
        all_vaf[i] = df['vaf']
        all_rmse[i] = df['rmse'][0]
        all_likelihood[i] = df['marginal_likeli'].item()

    # save data
    # get saving path
    path = path_general + 'data/'
    # to pandas
    all_df = pd.DataFrame(all_df)
    # filename
    file_name = file_name_general + '_gridsearch.csv'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # save data
    all_df.to_csv(path_general + file_name)
    # save performance values
    torch.save(all_vaf, path_general + 'data/' + 'all_vaf.pt')
    torch.save(all_rmse, path_general + 'data/' + 'all_rmse.pt')
    torch.save(all_likelihood, path_general + 'data/' + 'all_likelihood.pt')

    # plot performance
    dv.plot_perf_ndata(k_max_train_values, all_vaf, all_rmse, all_likelihood, options, path_general)

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))


# %%

if __name__ == "__main__":
    # set (high level) options dictionary
    options = {
        'dataset': 'narendra_li',  # only use this dynamic system here!
        'model': 'VRNN-Gauss',
        'do_train': True,
        'do_test': True,
        'logdir': 'ndata',
        'normalize': False,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': True,
        'savefig': True,
    }

    # values of evaluation
    vary_data = {
        'k_max_train_values': np.ceil(np.logspace(np.log10(5e3), np.log10(10e3), 2)).astype(int),
        'k_max_val_values': (5000. * np.ones(2)).astype(int),
        'k_max_test_values': (5000. * np.ones(2)).astype(int),
    }

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'], )
    # get saving file names
    file_name_general = options['dataset']

    # chosen values for evaluation
    params = {
        'h_best': 80,
        'z_best': 10,
        'n_best': 1,
    }

    run_main_ndata(options, vary_data, path_general, file_name_general, params)
