import os
from main_gridsearch import run_main_gridsearch

# set (high level) options dictionary
options = {
    'dataset': 'narendra_li',  # 'f16gvt', 'cascaded_tank', 'narendra_li'
    'model': 'VRNN-Gauss',
    'do_train': True,
    'do_test': True,
    'logdir': 'gridsearch',
    'normalize': True,
    'seed': 1235,
    'optim': 'Adam',
    'showfig': False,
    'savefig': True,
}

# select parameters for narendra-li benchmark
kwargs = {"k_max_train": 50000,
          "k_max_val": 5000,
          "k_max_test": 5000}

# values for grid search
gridvalues = {
    'h_values': [50, 60, 70],
    'z_values': [10, 15, 20],
    'n_values': [1],
}

"""'h_values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
# [40, 50]  # [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
'z_values': [1, 2, 5, 10, 15],  # [2]  # [1, 2, 5, 10, 15, 20]
'n_values': [1],  # [1]  # [1, 2, 5, 10]"""

# get saving path
path_general = os.getcwd() + '/log/{}/{}/run200222_50000/{}/'.format(options['logdir'],
                                                                     options['dataset'],
                                                                     options['model'], )
# get saving file names
file_name_general = options['dataset']

# %%
run_main_gridsearch(options, kwargs, gridvalues, path_general, file_name_general)
