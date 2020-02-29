import os
from main_gridsearch import run_main_gridsearch

# set (high level) options dictionary
options = {
    'dataset': 'wiener_hammerstein',  # 'f16gvt', 'cascaded_tank', 'narendra_li', 'toy_lgssm'
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

# select parameters for toy lgssm
kwargs = {}

# values for grid search
gridvalues = {
    'h_values': [30, 40, 50, 60],  # [10, 20, 30, 40, 50, 60, 70, 80],
    'z_values': [5, 10, 15],
    'n_values': [1],
}

adlog = 'run_200226'

# get saving path
path_general = os.getcwd() + '/log/{}/{}/{}/{}/'.format(options['logdir'],
                                                        options['dataset'],
                                                        adlog,
                                                        options['model'], )
# get saving file names
file_name_general = options['dataset']

# %%
run_main_gridsearch(options, kwargs, gridvalues, path_general, file_name_general)
