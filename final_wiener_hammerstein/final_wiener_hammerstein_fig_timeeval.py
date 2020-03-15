# import generic libraries
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

os.chdir('../')
sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import utils.dataevaluater as de
from utils.utils import compute_normalizer
# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState

# set (high level) options dictionary
options = {
    'dataset': 'wiener_hammerstein',
    'model': 'STORN',
    'do_train': False,
    'do_test': True,  # ALWAYS
    'logdir': 'final',
    'normalize': True,
    'seed': 1234,
    'optim': 'Adam',
    'showfig': True,
    'savefig': False,
}

test_set = 'sweptsine'
# get saving path
addlog = 'run_0310_hvar'
path_general = os.getcwd() + '/log_Server/{}/{}/{}/{}/'.format(options['logdir'],
                                                               options['dataset'],
                                                               addlog,
                                                               options['model'], )

# %%
if __name__ == "__main__":
    print('Run file: final_toy_lgssm.py')

    # get correct computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('Device: {}'.format(device))

    # get the options
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['test_options'] = train_params.get_test_options()

    # optimal model parameters
    h_opt = 50  # 60
    z_opt = 5  # 5
    n_opt = 3
    options['model_options'].h_dim = h_opt
    options['model_options'].z_dim = z_opt
    options['model_options'].n_layers = n_opt

    path = path_general + 'data/'
    MC_chosen = 0
    file_name_general = '{}_h{}_z{}_n{}_MC{}'.format(options['dataset'], h_opt, z_opt, n_opt, MC_chosen)

    # sampling period
    fs = 78125
    maxN = 4000
    kwargs = {'test_set': test_set}

    # Specifying datasets
    loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"].batch_size,
                                  test_batch_size=options["test_options"].batch_size,
                                  **kwargs)

    if options['do_test']:
        # %% test the model

        # ##### Loading the model -> This needs to be changed to run for GPU as well!!
        # switch to cpu computations for testing
        options['device'] = 'cpu'

        # Compute normalizers
        if options["normalize"]:
            normalizer_input, normalizer_output = compute_normalizer(loaders['test'])
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

        # load model
        path = path_general + 'model/'
        file_name = file_name_general + '_bestModel.ckpt'
        modelstate.load_model(path, file_name)
        modelstate.model.to(options['device'])

        # sample from the model
        for i, (u_test, y_test) in enumerate(loaders['test']):
            # getting output distribution parameter only implemented for selected models
            u_test = u_test.to(options['device'])
            u_test = u_test[:, :, :maxN]
            y_test = y_test[:, :, :maxN]
            y_sample, y_sample_mu, y_sample_sigma = modelstate.model.generate(u_test)

            # convert to numpy for evaluation
            # samples data
            y_sample_mu = y_sample_mu.detach().numpy()
            y_sample_sigma = y_sample_sigma.detach().numpy()
            # test data
            y_test = y_test.detach().numpy()
            y_sample = y_sample.detach().numpy()

        # %% plot time evaluation with uncertainty

        # plot resulting prediction
        plt.figure(1, figsize=(6, 5))
        length = y_test.shape[-1]
        x = np.linspace(0, length - 1, length) / fs
        # ####### plot true output
        plt.plot(x, y_test.squeeze(), label='{}'.format('Test Data, $\mu\pm3\sigma$'))

        # ####### plot samples output with uncertainty
        mean = y_sample_mu.squeeze()
        std = y_sample_sigma.squeeze()
        # plot mean
        plt.plot(x, mean, label='STORN, $\mu\pm3\sigma$')  # , color='r')
        # plot 3std around
        plt.fill_between(x, mean, mean + 3 * std, alpha=0.3, facecolor='r')
        plt.fill_between(x, mean, mean - 3 * std, alpha=0.3, facecolor='r')

        # #### plot settings
        plt.title('Output of WH')
        plt.ylabel('$y(k)$')
        plt.xlabel('time steps $k$')
        plt.legend()
        # plt.xlim([0, 0.055])
        # plt.ylim([-9, 9])

        # storage path
        file_name = '4_WH_results_STORN_{}'.format(kwargs['test_set'])
        path = path_general + 'timeEval/'
        if options['savefig']:
            # save figure
            # check if path exists and create otherwise
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + file_name + '.pdf', format='pdf')
        # plot model
        if options['showfig']:
            plt.show()
        plt.close(1)

        # %% test parameter

        print('Performance parameter of NN model:')
        # compute marginal likelihood (same as for predictive distribution loss in training)
        logLikelihood = de.compute_marginalLikelihood(y_test, y_sample_mu, y_sample_sigma, doprint=True)

        # compute VAF
        vaf = de.compute_vaf(y_test, y_sample_mu, doprint=True)

        # compute RMSE
        rmse = de.compute_rmse(y_test, y_sample_mu, doprint=True)

        # %% store simulation data in csv file
        muTest = y_test.squeeze()
        muModel = y_sample_mu.squeeze()
        p3sigmaModel = muModel + 3 * y_sample_sigma.squeeze()
        m3sigmaModel = muModel - 3 * y_sample_sigma.squeeze()

        data = {
            'x': x,
            'muTest': muTest,
            'muModel': muModel,
            'p3sigmaModel': p3sigmaModel,
            'm3sigmaModel': m3sigmaModel,
        }
        df = pd.DataFrame(data)  # , columns=['muTest', 'sigmaTest'])

        path = os.getcwd() + '/final_wiener_hammerstein/' + 'WH_data_{}_timeeval.csv'.format(kwargs['test_set'])
        df.to_csv(path, index=False)
