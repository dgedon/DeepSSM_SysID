import os
from main_gridsearch import run_main_gridsearch

# set (high level) options dictionary
options = {
    'dataset': 'toy_lgssm',  # 'f16gvt', 'cascaded_tank', 'narendra_li', 'toy_lgssm'
    'model': 'VRNN-GMM',
    'do_train': True,
    'do_test': True,
    'logdir': 'gridsearch',
    'normalize': True,
    'seed': 1234,
    'optim': 'Adam',
    'showfig': False,
    'savefig': True,
}

# select parameters for toy lgssm
kwargs = {"k_max_train": 2000,
          "k_max_val": 2000,
          "k_max_test": 5000}

# values for grid search
gridvalues = {
    'h_values': [50, 60, 70, 80],
    # [40, 50]  # [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    'z_values': [2, 5, 10],  # [2]  # [1, 2, 5, 10, 15, 20]
    'n_values': [1],  # [1]  # [1, 2, 5, 10]
}

adlog = 'run_200221'

# get saving path
path_general = os.getcwd() + '/log/{}/{}/{}/{}/'.format(options['logdir'],
                                                        options['dataset'],
                                                        adlog,
                                                        options['model'], )
# get saving file names
file_name_general = options['dataset']

# %%
run_main_gridsearch(options, kwargs, gridvalues, path_general, file_name_general)
