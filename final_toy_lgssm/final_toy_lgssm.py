# import generic libraries
import pandas as pd
import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

os.chdir('../')
sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import models.model_state
import training
import utils.dataevaluater as de
import utils.datavisualizer as dv
from utils.utils import get_n_params
from utils.kalman_filter import run_kalman_filter
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
# import options files
import options.model_options as model_params
import options.dataset_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState

# set (high level) options dictionary
options = {
    'dataset': 'toy_lgssm',
    'model': 'STORN',
    'do_train': False,
    'do_test': True,  # ALWAYS
    'logdir': 'final',
    'normalize': True,
    'seed': 1234,
    'optim': 'Adam',
    'showfig': True,
    'savefig': True,
    'MCsamples': 50,
    'optValue': {
        'h_opt': 60,
        'z_opt': 5,
        'n_opt': 1, },
}

# get saving path
path_general = os.getcwd() + '/log_Server/{}/{}/{}/'.format(options['logdir'],
                                                            options['dataset'],
                                                            options['model'], )

# %%
if __name__ == "__main__":
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, options['dataset'] + '_runlog')

    start_time = time.time()
    print('Run file: final_toy_lgssm.py')
    print(time.strftime("%c"))

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
    options['model_options'].h_dim = options['optValue']['h_opt']
    options['model_options'].z_dim = options['optValue']['z_opt']
    options['model_options'].n_layers = options['optValue']['n_opt']

    # save the options
    save_options(options, path_general, 'options.txt')

    # allocation
    rmse_all = np.zeros([options['MCsamples']])
    vaf_all = np.zeros([options['MCsamples']])
    logLikelihood_all = np.zeros([options['MCsamples']])
    rmse_KF_all = np.zeros([options['MCsamples']])
    vaf_KF_all = np.zeros([options['MCsamples']])

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    file_name_general = '{}_h{}_z{}_n{}'.format(options['dataset'],
                                                options['model_options'].h_dim,
                                                options['model_options'].z_dim,
                                                options['model_options'].n_layers)

    # %% Monte Carlo runs

    for mcIter in range(options['MCsamples']):
        print('\n#####################')
        print('MC ITERATION: {}/{}'.format(mcIter+1, options['MCsamples']))
        print('#####################\n')

        # set the correct device to run on
        options['device'] = device

        file_name_general_it = file_name_general + '_MC{}'.format(mcIter)

        # select parameters for toy lgssm
        kwargs = {"k_max_train": 2000,
                  "k_max_val": 2000,
                  "k_max_test": 5000}

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

        df = {}
        if options['do_train']:
            # %% train the model
            df = training.run_train(modelstate=modelstate,
                                    loader_train=loaders['train'],
                                    loader_valid=loaders['valid'],
                                    options=options,
                                    dataframe={},
                                    path_general=path_general,
                                    file_name_general=file_name_general_it)

        if options['do_test']:
            # %% test the model

            # ##### Loading the model -> This needs to be changed to run for GPU as well!!
            # switch to cpu computations for testing
            options['device'] = 'cpu'

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

            # load model
            path = path_general + 'model/'
            file_name = file_name_general_it + '_bestModel.ckpt'
            modelstate.load_model(path, file_name)
            modelstate.model.to(options['device'])

            # plot and save the loss curve
            dv.plot_losscurve(df, options, path_general, file_name_general_it, removedata=False)

            # sample from the model
            for i, (u_test, y_test) in enumerate(loaders['test']):
                # getting output distribution parameter only implemented for selected models
                u_test = u_test.to(options['device'])
                y_sample, y_sample_mu, y_sample_sigma = modelstate.model.generate(u_test)

                # convert to numpy for evaluation
                # samples data
                y_sample_mu = y_sample_mu.detach().numpy()
                y_sample_sigma = y_sample_sigma.detach().numpy()
                # test data
                y_test = y_test.detach().numpy()
                y_sample = y_sample.detach().numpy()

            # original test set is unnoisy -> get noisy test set
            yshape = y_test.shape
            y_test_noisy = y_test + np.sqrt(1) * np.random.randn(yshape[0], yshape[1], yshape[2])

            # run Kalman filter as optimal estimator for LGSSM
            A = np.array([[0.7, 0.8], [0, 0.1]])
            B = np.array([[-1], [0.1]])
            C = np.array([[1], [0]]).transpose()
            Q = np.sqrt(0.25) * np.identity(2)
            R = np.sqrt(1) * np.identity(1)
            y_kalman = run_kalman_filter(A, B, C, Q, R, u_test, y_test_noisy)

            # %% plot time evaluation with uncertainty

            # plot resulting prediction
            data_y_true = [y_test, np.sqrt(1) * np.ones_like(y_test)]
            data_y_sample = [y_sample_mu, y_sample_sigma]
            label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']

            plt.figure(figsize=(5, 5))
            length = y_test.shape[-1]
            x = np.linspace(0, length - 1, length)
            # ####### plot true output with uncertainty
            mean = y_test.squeeze()
            std = np.sqrt(1) * np.ones_like(mean)
            # plot mean
            plt.plot(mean, label='y_1(k) {}'.format('true, $\mu\pm3\sigma$'))
            # plot 3std around
            plt.fill_between(x, mean, mean + 3 * std, alpha=0.3, facecolor='b')
            plt.fill_between(x, mean, mean - 3 * std, alpha=0.3, facecolor='b')

            # ####### plot KF output
            plt.plot(x, y_kalman.squeeze(), label='y_1(k) Kalman filter', linestyle='dashed', color='k')

            # ####### plot samples output with uncertainty
            mean = y_sample_mu.squeeze()
            std = y_sample_sigma.squeeze()
            # plot mean
            plt.plot(mean, label='y_1(k) {}'.format('sample, $\mu\pm3\sigma$'))
            # plot 3std around
            plt.fill_between(x, mean, mean + 3 * std, alpha=0.3, facecolor='r')
            plt.fill_between(x, mean, mean - 3 * std, alpha=0.3, facecolor='r')

            # #### plot settings
            plt.title('Output $y_1(k)$, {} with (h,z,n)=({},{},{})'.format(options['dataset'],
                                                                           options['model_options'].h_dim,
                                                                           options['model_options'].z_dim,
                                                                           options['model_options'].n_layers))

            plt.ylabel('$y_1(k)$')
            plt.xlabel('time steps $k$')
            plt.legend()
            plt.xlim([0, 100])

            # storage path
            file_name = file_name_general_it + '_timeEval'
            path = path_general + 'timeEval/'
            if options['savefig']:
                # save figure
                # check if path exists and create otherwise
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path + file_name + '.png', format='png')
            # plot model
            if options['showfig']:
                plt.show()

            # %% test parameter

            print('Performance parameter of NN model:')
            # compute marginal likelihood (same as for predictive distribution loss in training)
            logLikelihood = de.compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)

            # compute VAF
            vaf = de.compute_vaf(y_test_noisy, y_sample_mu, doprint=True)

            # compute RMSE
            rmse = de.compute_rmse(y_test_noisy, y_sample_mu, doprint=True)

            print('\nPerformance parameter of KF:')
            # compute VAF
            vaf_KF = de.compute_vaf(y_test_noisy, np.expand_dims(y_kalman, 0), doprint=True)
            # compute RMSE
            rmse_KF = de.compute_rmse(y_test_noisy, np.expand_dims(y_kalman, 0), doprint=True)

            # %% performance parameters saving
            vaf_all[mcIter] = vaf
            rmse_all[mcIter] = rmse
            logLikelihood_all[mcIter] = logLikelihood
            vaf_KF_all[mcIter] = vaf_KF
            rmse_KF_all[mcIter] = rmse_KF

    # %% print mean evaluation values

    print('\nModel: mean VAF = {}'.format(vaf_all.mean()))
    print('Model: mean RMSE = {}'.format(rmse_all.mean()))
    print('Model: mean log Likelihood = {}'.format(logLikelihood_all.mean()))

    print('\nKF: mean VAF = {}'.format(vaf_KF_all.mean()))
    print('KF: mean RMSE = {}'.format(rmse_KF_all.mean()))

    # %% save data
    # get saving path
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # filename
    file_name = file_name_general + '.pt'
    # data to save
    datasaver = {'vaf_all': vaf_all,
                 'rmse_all': rmse_all,
                 'logLikelihood_all': logLikelihood_all,
                 'vaf_KF_all': vaf_KF_all,
                 'rmse_KF_all': rmse_KF_all,
                 'df': df}
    # save data
    torch.save(datasaver, path + file_name)

    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))
