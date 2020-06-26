# import generic libraries
import torch.utils.data
import pandas as pd
import os
import torch
import time
import sys
import matplotlib.pyplot as plt

os.chdir('../')
sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_single(options, path_general, file_name_general):
    start_time = time.time()
    print('Run file: main_single.py')
    print(time.strftime("%c"))

    # get correct computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
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

    file_name_general = file_name_general + '_h{}_z{}_n{}'.format(options['model_options'].h_dim,
                                                                  options['model_options'].z_dim,
                                                                  options['model_options'].n_layers)
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general)

    # Specifying datasets
    loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"].batch_size,
                                  test_batch_size=options["test_options"].batch_size, )

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

    # save the options
    save_options(options, path_general, 'options.txt')

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
                                file_name_general=file_name_general)

    if options['do_test']:
        # test the model
        df = testing.run_test(options, loaders, df, path_general, file_name_general)

    # save data
    # get saving path
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # to pandas
    df = pd.DataFrame(df)
    # filename
    file_name = file_name_general + '.csv'
    # save data
    df.to_csv(path + file_name)

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))


# %%
if __name__ == "__main__":
    # set (high level) options dictionary
    options = {
        'dataset': 'toy_lgssm',  # options: 'narendra_li', 'toy_lgssm', 'wiener_hammerstein'
        'model': 'STORN', # options: 'VAE-RNN', 'VRNN-Gauss', 'VRNN-Gauss-I', 'VRNN-GMM', 'VRNN-GMM-I', 'STORN'
        'do_train': True,
        'do_test': True,
        'logdir': 'single',
        'normalize': True,
        'seed': 1234,
        'optim': 'Adam',
        'showfig': True,
        'savefig': False,
    }

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'], )

    # get saving file names
    file_name_general = options['dataset']

    run_main_single(options, path_general, file_name_general)
