
import os
import numpy as np
from main_ndata import run_main_ndata



# %%####################################################################################################################
# Main function
########################################################################################################################


# %%

if __name__ == "__main__":
    # set (high level) options dictionary
    options = {
        'dataset': 'narendra_li',  # only use this dynamic system here!
        'model': 'VRNN-Gauss-I',
        'do_train': True,
        'do_test': True,
        'logdir': 'ndata',
        'normalize': False,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': False,
        'savefig': True,
    }

    # values of evaluation
    vary_data = {
        'k_max_train_values': np.array([2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000]),
        'k_max_val_values': (5000. * np.ones(8)).astype(int),
        'k_max_test_values': (5000. * np.ones(8)).astype(int),
    }

    add_log = 'run200223'

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}/{}/'.format(options['logdir'],
                                                            options['dataset'],
                                                            add_log,
                                                            options['model'], )
    # get saving file names
    file_name_general = options['dataset']

    # chosen values for evaluation
    params = {
        'h_best': 60,
        'z_best': 20,
        'n_best': 1,
    }

    run_main_ndata(options, vary_data, path_general, file_name_general, params)
